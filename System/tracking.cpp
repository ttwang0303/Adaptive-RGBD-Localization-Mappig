#include "tracking.h"
#include "Core/keyframe.h"
#include "Core/keyframedatabase.h"
#include "Core/landmark.h"
#include "Core/map.h"
#include "Features/extractor.h"
#include "Features/matcher.h"
#include "Odometry/odometry.h"
#include "Odometry/pnpransac.h"
#include "Odometry/pnpsolver.h"
#include "Odometry/ransac.h"
#include "Utils/converter.h"
#include "localmapping.h"

using namespace std;

Tracking::Tracking(DBoW3::Vocabulary* pVoc, Map* pMap, Database* pKFDB, Extractor* pExtractor)
    : mState(NO_IMAGES_YET)
    , mpVocabulary(pVoc)
    , mpKeyFrameDB(pKFDB)
    , mpMap(pMap)
    , mpExtractor(pExtractor)
{
}

void Tracking::SetLocalMapper(LocalMapping* pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetOdometer(Odometry* pOdometer)
{
    mpOdometer = pOdometer;
}

cv::Mat Tracking::Track(const cv::Mat& imColor, const cv::Mat& imDepth, const double& timestamp)
{
    mpCurrentFrame.reset(new Frame(imColor, imDepth, timestamp));
    mpCurrentFrame->ExtractFeatures(mpExtractor);

    if (mState == NO_IMAGES_YET)
        mState = NOT_INITIALIZED;

    mLastProcessedState = mState;
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED) {
        Initialization();
    } else {
        bool bOK;
        if (mState == OK) {
            CheckReplaced();

            if (mVelocity.empty())
                bOK = TrackReferenceKF();
            else
                bOK = TrackModel();
        } else
            bOK = false;

        mpCurrentFrame->mpReferenceKF = mpReferenceKF;

        if (bOK) {
            if (TrackLocalMap()) {
                mState = OK;
                UpdateMotionModel();
                CleanVOmatches();
                DeleteTemporalPoints();

                if (NeedNewKF())
                    CreateNewKF();

                for (size_t i = 0; i < mpCurrentFrame->N; ++i) {
                    if (mpCurrentFrame->GetLandmark(i) && mpCurrentFrame->IsOutlier(i))
                        mpCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
                }

            } else {
                mState = LOST;
                if (mpMap->KeyFramesInMap() <= 5) {
                    cerr << "Track lost soon after initialisation" << endl;
                    terminate();
                }
            }

            if (!mpCurrentFrame->mpReferenceKF)
                mpCurrentFrame->mpReferenceKF = mpReferenceKF;

            // mpLastFrame = mpCurrentFrame;
        }

        mpLastFrame = mpCurrentFrame;
    }

    UpdateRelativePose();
}

void Tracking::Initialization()
{
    if (mpCurrentFrame->N < 400) {
        cerr << "mCurrentFrame.N: " << mpCurrentFrame->N << " < 500" << endl;
        terminate();
    }

    mpCurrentFrame->SetPose(cv::Mat::eye(4, 4, CV_32F));

    KeyFrame* pKFini = new KeyFrame(*mpCurrentFrame, mpMap, mpKeyFrameDB);
    mpMap->AddKeyFrame(pKFini);

    // Create Landmarks and associate to KeyFrame
    for (size_t i = 0; i < mpCurrentFrame->N; ++i) {
        const float& z = mpCurrentFrame->mvKeys3Dc[i].z;
        if (z > 0) {
            cv::Mat x3Dw = mpCurrentFrame->UnprojectWorld(i);
            Landmark* pLM = new Landmark(x3Dw, pKFini, mpMap);
            pLM->AddObservation(pKFini, i);
            pKFini->AddLandmark(pLM, i);
            pLM->ComputeDistinctiveDescriptors();
            mpMap->AddLandmark(pLM);

            mpCurrentFrame->AddLandmark(pLM, i);
        }
    }

    cout << "New map created with " << mpMap->LandmarksInMap() << " landmarks" << endl;

    mpLocalMapper->InsertKeyFrame(pKFini);

    mpLastFrame = mpCurrentFrame;
    mnLastKFid = mpCurrentFrame->mnId;
    mpLastKF = pKFini;

    mvpLocalKFs.push_back(pKFini);
    mvpLocalLMs = mpMap->GetAllLandmarks();
    mpReferenceKF = pKFini;
    mpCurrentFrame->mpReferenceKF = pKFini;

    mState = OK;
}

void Tracking::CheckReplaced()
{
    for (size_t i = 0; i < mpLastFrame->N; ++i) {
        Landmark* pLM = mpLastFrame->GetLandmark(i);

        if (pLM) {
            Landmark* pRep = pLM->GetReplaced();
            if (pRep)
                mpLastFrame->AddLandmark(pRep, i);
        }
    }
}

bool Tracking::TrackReferenceKF()
{
    Matcher matcher(0.7f);
    vector<cv::DMatch> vMatches12;
    size_t nmatches = matcher.KnnMatch(mpReferenceKF, *mpCurrentFrame, vMatches12);

    if (nmatches < 15)
        return false;

    mpOdometer->Compute(mpReferenceKF, mpCurrentFrame.get(), vMatches12);
    int inliers = DiscardOutliers();
    Matcher::DrawInlierPoints(*mpCurrentFrame);
    return inliers >= 10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mpLastFrame->mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    mpLastFrame->SetPose(Tlr * pRef->GetPose());

    if (mnLastKFid == mpLastFrame->mnId)
        return;

    // Create "visual odometry" Landmarks
    // We sort points according to their measured depth
    vector<pair<float, size_t>> vDepthIdx;
    vDepthIdx.reserve(mpLastFrame->N);
    for (size_t i = 0; i < mpLastFrame->N; i++) {
        float z = mpLastFrame->mvKeys3Dc[i].z;
        if (z > 0) {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<3.09m)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++) {
        size_t i = vDepthIdx[j].second;

        bool bCreateNew = false;

        Landmark* pLM = mpLastFrame->GetLandmark(i);
        if (!pLM)
            bCreateNew = true;
        else if (pLM->Observations() < 1) {
            bCreateNew = true;
        }

        if (bCreateNew) {
            cv::Mat x3D = mpLastFrame->UnprojectWorld(i);
            Landmark* pNewLM = new Landmark(x3D, mpMap, mpLastFrame.get(), i);

            mpLastFrame->AddLandmark(pNewLM, i);

            mlpTemporalPoints.push_back(pNewLM);
            nPoints++;
        } else {
            nPoints++;
        }

        if (vDepthIdx[j].first > 3.09f && nPoints > 100)
            break;
    }
}

bool Tracking::TrackModel()
{
    UpdateLastFrame();
    mpCurrentFrame->SetPose(mVelocity * mpLastFrame->GetPose());

    Matcher matcher(0.9f);
    vector<cv::DMatch> vMatches12;
    size_t nmatches = matcher.KnnMatch(*mpLastFrame, *mpCurrentFrame, vMatches12);

    if (nmatches < 20)
        return false;

    //mpOdometer->Compute(mpLastFrame.get(), mpCurrentFrame.get(), vMatches12);
    PnPRansac::Compute(*mpCurrentFrame);
    int inliers = DiscardOutliers();
    return inliers >= 10;
}

int Tracking::DiscardOutliers()
{
    int nmatchesMap = 0;
    for (size_t i = 0; i < mpCurrentFrame->N; i++) {
        Landmark* pLM = mpCurrentFrame->GetLandmark(i);
        if (!pLM)
            continue;

        if (mpCurrentFrame->IsOutlier(i)) {
            mpCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
            mpCurrentFrame->SetInlier(i);
            pLM->mbTrackInView = false;
            pLM->mnLastFrameSeen = mpCurrentFrame->mnId;
        } else /*if (pLM->Observations() > 0)*/ {
            nmatchesMap++;
        }
    }

    return nmatchesMap;
}

bool Tracking::TrackLocalMap()
{
    // Update
    UpdateLocalKFs();
    UpdateLocalLMs();

    SearchLocalLMs();

    // PnPSolver::Compute(mpCurrentFrame.get());
    PnPRansac::Compute(*mpCurrentFrame);
    Matcher::DrawInlierPoints(*mpCurrentFrame);

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (size_t i = 0; i < mpCurrentFrame->N; i++) {
        if (mpCurrentFrame->GetLandmark(i)) {
            if (mpCurrentFrame->IsInlier(i)) {
                mpCurrentFrame->GetLandmark(i)->IncreaseFound();
                if (mpCurrentFrame->GetLandmark(i)->Observations() > 0)
                    mnMatchesInliers++;
            }
        }
    }

    if (mnMatchesInliers < 5)
        return false;
    else
        return true;
}

void Tracking::UpdateLocalKFs()
{
    // Each MapPoint vote for the keyframes in which it has been observed
    map<KeyFrame*, int> keyframeCounter;
    for (size_t i = 0; i < mpCurrentFrame->N; i++) {
        Landmark* pLM = mpCurrentFrame->GetLandmark(i);
        if (pLM) {
            if (!pLM->isBad()) {
                const map<KeyFrame*, size_t> observations = pLM->GetObservations();
                for (const auto& [pKF, idx] : observations)
                    keyframeCounter[pKF]++;
            } else {
                mpCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame* pKFmax = static_cast<KeyFrame*>(nullptr);

    mvpLocalKFs.clear();
    mvpLocalKFs.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map.
    // Also check which keyframe shares most points
    for (const auto& [pKF, nObs] : keyframeCounter) {
        if (pKF->isBad())
            continue;

        if (nObs > max) {
            max = nObs;
            pKFmax = pKF;
        }

        mvpLocalKFs.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (const auto pLocalKF : mvpLocalKFs) {
        // Limit the number of keyframes
        if (mvpLocalKFs.size() > 80)
            break;

        const vector<KeyFrame*> vNeighs = pLocalKF->GetBestCovisibilityKeyFrames(10);
        for (auto pNeighKF : vNeighs) {
            if (!pNeighKF->isBad()) {
                if (pNeighKF->mnTrackReferenceForFrame != mpCurrentFrame->mnId) {
                    mvpLocalKFs.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pLocalKF->GetChilds();
        for (auto pChildKF : spChilds) {
            if (!pChildKF->isBad()) {
                if (pChildKF->mnTrackReferenceForFrame != mpCurrentFrame->mnId) {
                    mvpLocalKFs.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pLocalKF->GetParent();
        if (pParent) {
            if (pParent->mnTrackReferenceForFrame != mpCurrentFrame->mnId) {
                mvpLocalKFs.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
                break;
            }
        }
    }

    if (pKFmax) {
        mpReferenceKF = pKFmax;
        mpCurrentFrame->mpReferenceKF = mpReferenceKF;
    }
}

void Tracking::UpdateLocalLMs()
{
    mvpLocalLMs.clear();

    for (auto pKF : mvpLocalKFs) {
        const vector<Landmark*> vpLMs = pKF->GetLandmarks();

        for (auto pLM : vpLMs) {
            if (!pLM)
                continue;
            if (pLM->mnTrackReferenceForFrame == mpCurrentFrame->mnId)
                continue;
            if (!pLM->isBad()) {
                mvpLocalLMs.push_back(pLM);
                pLM->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
            }
        }
    }
}

void Tracking::SearchLocalLMs()
{
    // Do not search map points already matched
    vector<Landmark*> vpLMs = mpCurrentFrame->GetLandmarks();
    for (auto pLM : vpLMs) {
        if (pLM) {
            if (pLM->isBad()) {
                pLM = static_cast<Landmark*>(nullptr);
            } else {
                pLM->IncreaseVisible();
                pLM->mnLastFrameSeen = mpCurrentFrame->mnId;
                pLM->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (auto pLM : mvpLocalLMs) {
        if (pLM->mnLastFrameSeen == mpCurrentFrame->mnId)
            continue;
        if (pLM->isBad())
            continue;

        // Project (this fills MapPoint variables for matching)
        if (mpCurrentFrame->isInFrustum(pLM)) {
            pLM->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0) {
        Matcher matcher(0.8f);
        matcher.ProjectionMatch(mpCurrentFrame.get(), mvpLocalLMs, 5.0f);
    }
}

void Tracking::UpdateRelativePose()
{
    cv::Mat Tcr = mpCurrentFrame->GetPose() * mpCurrentFrame->mpReferenceKF->GetPoseInv();
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(mpCurrentFrame->mTimestamp);
}

void Tracking::UpdateMotionModel()
{
    if (!mpLastFrame->GetPose().empty())
        mVelocity = mpCurrentFrame->GetPose() * mpLastFrame->GetPoseInv();
    else
        mVelocity = cv::Mat();
}

void Tracking::CleanVOmatches()
{
    for (size_t i = 0; i < mpCurrentFrame->N; ++i) {
        Landmark* pLM = mpCurrentFrame->GetLandmark(i);
        if (!pLM)
            continue;
        if (pLM->Observations() < 1) {
            mpCurrentFrame->SetInlier(i);
            mpCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
        }
    }
}

void Tracking::DeleteTemporalPoints()
{
    for (list<Landmark*>::iterator lit = mlpTemporalPoints.begin(); lit != mlpTemporalPoints.end(); lit++) {
        Landmark* pLM = *lit;
        delete pLM;
    }
    mlpTemporalPoints.clear();
}

double Tracking::tNorm(const cv::Mat& T)
{
    cv::Mat t = T.rowRange(0, 3).col(3);
    return cv::norm(t);
}

double Tracking::RNorm(const cv::Mat& T)
{
    cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
    return acos(0.5 * (R.at<float>(0, 0) + R.at<float>(1, 1) + R.at<float>(2, 2) - 1.0));
}

bool Tracking::NeedNewKF()
{
    static const double mint = 0.15;
    static const double minR = 0.15;

    if (mpLocalMapper->isStopped() || mpLocalMapper->StopRequested())
        return false;

    bool bLocalMappingIdle = mpLocalMapper->AcceptKFs();

    cv::Mat delta = mpCurrentFrame->GetPoseInv() * mpReferenceKF->GetPose();
    bool c1 = (tNorm(delta) > mint || RNorm(delta) > minR);

    if (c1) {
        if (bLocalMappingIdle) {
            return true;
        } else {
            mpLocalMapper->InterruptBA();
            if (mpLocalMapper->KeyFramesInQueue() < 3)
                return true;
            else {
                return false;
            }
        }
    } else {
        return false;
    }
}

void Tracking::CreateNewKF()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(*mpCurrentFrame, mpMap, mpKeyFrameDB);

    mpReferenceKF = pKF;
    mpCurrentFrame->mpReferenceKF = pKF;

    mpCurrentFrame->UpdatePoseMatrices();

    // We sort points by the measured depth by the RGBD sensor.
    // We create all those Landmarks whose depth < 3.09m.
    // If there are less than 100 close points we create the 100 closest.
    vector<pair<float, size_t>> vDepthIdx;
    vDepthIdx.reserve(mpCurrentFrame->N);
    for (size_t i = 0; i < mpCurrentFrame->N; i++) {
        float z = mpCurrentFrame->mvKeys3Dc[i].z;
        if (z > 0)
            vDepthIdx.push_back(make_pair(z, i));
    }

    if (!vDepthIdx.empty()) {
        sort(vDepthIdx.begin(), vDepthIdx.end());

        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            size_t i = vDepthIdx[j].second;

            bool bCreateNew = false;

            Landmark* pMP = mpCurrentFrame->GetLandmark(i);
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                bCreateNew = true;
                mpCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
            }

            if (bCreateNew) {
                cv::Mat x3D = mpCurrentFrame->UnprojectWorld(i);
                Landmark* pNewLM = new Landmark(x3D, pKF, mpMap);
                pNewLM->AddObservation(pKF, i);
                pKF->AddLandmark(pNewLM, i);
                pNewLM->ComputeDistinctiveDescriptors();
                mpMap->AddLandmark(pNewLM);

                mpCurrentFrame->AddLandmark(pNewLM, i);
                nPoints++;
            } else {
                nPoints++;
            }

            if (vDepthIdx[j].first > 3.09f && nPoints > 100)
                break;
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);
    mpLocalMapper->SetNotStop(false);

    mnLastKFid = mpCurrentFrame->mnId;
    mpLastKF = pKF;
}

void Tracking::SaveTrajectory(const string& filename)
{
    cout << "Saving camera trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    cv::Mat Two = vpKFs[0]->GetPoseInv();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    list<KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<double>::iterator lT = mlFrameTimes.begin();
    for (list<cv::Mat>::iterator lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++) {
        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

        while (pKF->isBad()) {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        cv::Mat Tcw = (*lit) * Trw;
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << " -Camera trajectory saved!" << endl;
}

void Tracking::SaveKeyFrameTrajectory(const string filename)
{
    cout << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for (const auto pKF : vpKFs) {
        if (pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimestamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }

    f.close();
    cout << " -KeyFrame trajectory saved!" << endl;
}
