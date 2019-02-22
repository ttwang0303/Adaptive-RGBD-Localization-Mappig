#include "tracking.h"
#include "Core/keyframe.h"
#include "Core/keyframedatabase.h"
#include "Core/landmark.h"
#include "Core/map.h"
#include "Features/extractor.h"
#include "Features/matcher.h"
#include "Odometry/odometry.h"
#include "Odometry/ransac.h"

using namespace std;

Tracking::Tracking(DBoW3::Vocabulary* pVoc, Map* pMap, Database* pKFDB, Extractor* pExtractor)
    : mState(NO_IMAGES_YET)
    , mpVocabulary(pVoc)
    , mpKeyFrameDB(pKFDB)
    , mpMap(pMap)
    , mpExtractor(pExtractor)
{
}

void Tracking::SetOdometer(Odometry* pOdometer)
{
    mpOdometer = pOdometer;
}

cv::Mat Tracking::Track(const cv::Mat& imColor, const cv::Mat& imDepth, const double& timestamp)
{
    mCurrentFrame.reset(new Frame(imColor, imDepth, timestamp));
    mCurrentFrame->ExtractFeatures(mpExtractor);

    if (mState == NO_IMAGES_YET)
        mState = NOT_INITIALIZED;

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

        mCurrentFrame->mpReferenceKF = mpReferenceKF;

        if (bOK) {
            if (TrackLocalMap()) {
                mState = OK;
                UpdateMotionModel();
                CleanVOmatches();
                DeleteTemporalPoints();

                if (NeedNewKF())
                    CreateNewKF();

                for (size_t i = 0; i < mCurrentFrame->N; ++i) {
                    if (mCurrentFrame->GetLandmark(i) && mCurrentFrame->IsOutlier(i))
                        mCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
                }

            } else {
                mState = LOST;
                if (mpMap->KeyFramesInMap() <= 5) {
                    cerr << "Track lost soon after initialisation" << endl;
                    terminate();
                }
            }

            if (!mCurrentFrame->mpReferenceKF)
                mCurrentFrame->mpReferenceKF = mpReferenceKF;

            mLastFrame = mCurrentFrame;
        }
    }
}

void Tracking::Initialization()
{
    if (mCurrentFrame->N < 500) {
        cerr << "mCurrentFrame.N: " << mCurrentFrame->N << " < 500" << endl;
        terminate();
    }

    mCurrentFrame->SetPose(cv::Mat::eye(4, 4, CV_32F));

    KeyFrame* pKFini = new KeyFrame(*mCurrentFrame, mpMap, mpKeyFrameDB);
    mpMap->AddKeyFrame(pKFini);

    // Create Landmarks and associate to KeyFrame
    for (size_t i = 0; i < mCurrentFrame->N; ++i) {
        const float& z = mCurrentFrame->mvKeys3Dc[i].z;
        if (z > 0) {
            cv::Mat x3Dw = mCurrentFrame->UnprojectWorld(i);
            Landmark* pLM = new Landmark(x3Dw, pKFini, mpMap);
            pLM->AddObservation(pKFini, i);
            pKFini->AddLandmark(pLM, i);
            pLM->ComputeDistinctiveDescriptors();
            mpMap->AddLandmark(pLM);

            mCurrentFrame->AddLandmark(pLM, i);
        }
    }

    cout << "New map created with " << mpMap->LandmarksInMap() << " landmarks" << endl;

    mLastFrame = mCurrentFrame;
    mnLastKFid = mCurrentFrame->GetId();
    mpLastKF = pKFini;

    mvpLocalKFs.push_back(pKFini);
    mvpLocalLMs = mpMap->GetAllLandmarks();
    mpReferenceKF = pKFini;
    mCurrentFrame->mpReferenceKF = pKFini;

    mState = OK;
}

void Tracking::CheckReplaced()
{
    for (size_t i = 0; i < mLastFrame->N; ++i) {
        Landmark* pLM = mLastFrame->GetLandmark(i);

        if (pLM) {
            Landmark* pRep = pLM->GetReplaced();
            if (pRep)
                mLastFrame->AddLandmark(pRep, i);
        }
    }
}

bool Tracking::TrackReferenceKF()
{
    Matcher matcher(0.7f);
    vector<cv::DMatch> vMatches12;
    size_t nmatches = matcher.KnnMatch(mpReferenceKF, *mCurrentFrame, vMatches12);

    if (nmatches < 15)
        return false;

    mpOdometer->Compute(mpReferenceKF, mCurrentFrame.get(), vMatches12);
    int inliers = DiscardOutliers();
    Matcher::DrawInlierPoints(*mCurrentFrame);
    return inliers >= 10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // KeyFrame* pRef = mLastFrame.mpReferenceKF;
    // cv::Mat Tlr = mlRelativeFramePoses.back();
    // mLastFrame.SetPose(Tlr * pRef->GetPose());

    if (mnLastKFid == mLastFrame->GetId())
        return;

    // Create "visual odometry" Landmarks
    // We sort points according to their measured depth
    vector<pair<float, size_t>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame->N);
    for (size_t i = 0; i < mLastFrame->N; i++) {
        float z = mLastFrame->mvKeys3Dc[i].z;
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

        Landmark* pLM = mLastFrame->GetLandmark(i);
        if (!pLM)
            bCreateNew = true;
        else if (pLM->Observations() < 1) {
            bCreateNew = true;
        }

        if (bCreateNew) {
            cv::Mat x3D = mLastFrame->UnprojectWorld(i);
            Landmark* pNewLM = new Landmark(x3D, mpMap, mLastFrame.get(), i);

            mLastFrame->AddLandmark(pNewLM, i);

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
    mCurrentFrame->SetPose(mVelocity * mLastFrame->GetPose());

    Matcher matcher(0.9f);
    vector<cv::DMatch> vMatches12;
    size_t nmatches = matcher.KnnMatch(*mLastFrame, *mCurrentFrame, vMatches12);

    if (nmatches < 20)
        return false;

    mpOdometer->Compute(mLastFrame.get(), mCurrentFrame.get(), vMatches12);
    int inliers = DiscardOutliers();
    Matcher::DrawInlierPoints(*mCurrentFrame);
    return inliers >= 10;
}

int Tracking::DiscardOutliers()
{
    int nmatchesMap = 0;
    for (size_t i = 0; i < mCurrentFrame->N; i++) {
        Landmark* pLM = mCurrentFrame->GetLandmark(i);
        if (!pLM)
            continue;

        if (mCurrentFrame->IsOutlier(i)) {
            mCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
            mCurrentFrame->SetInlier(i);
            pLM->mbTrackInView = false;
            pLM->mnLastFrameSeen = mCurrentFrame->GetId();
        } else /*if (pLM->Observations() > 0)*/ {
            nmatchesMap++;
        }
    }

    return nmatchesMap;
}

bool Tracking::TrackLocalMap()
{
    return true;
}

void Tracking::UpdateMotionModel()
{
    if (!mLastFrame->GetPose().empty())
        mVelocity = mCurrentFrame->GetPose() * mLastFrame->GetPoseInv();
    else
        mVelocity = cv::Mat();
}

void Tracking::CleanVOmatches()
{
    for (size_t i = 0; i < mCurrentFrame->N; ++i) {
        Landmark* pLM = mCurrentFrame->GetLandmark(i);
        if (!pLM)
            continue;
        if (pLM->Observations() < 1) {
            mCurrentFrame->SetInlier(i);
            mCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
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
    static const double mint = 0.25;
    static const double minR = 0.25;

    cv::Mat delta = mCurrentFrame->GetPoseInv() * mpReferenceKF->GetPose();
    return (tNorm(delta) > mint || RNorm(delta) > minR);
}

void Tracking::CreateNewKF()
{
    KeyFrame* pKF = new KeyFrame(*mCurrentFrame, mpMap, mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame->mpReferenceKF = pKF;
    mpMap->AddKeyFrame(pKF);
    cout << "New KF" << endl;

    // mCurrentFrame.UpdatePoseMatrices();

    // We sort points by the measured depth by the RGBD sensor.
    // We create all those Landmarks whose depth < 3.09m.
    // If there are less than 100 close points we create the 100 closest.
    vector<pair<float, size_t>> vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame->N);
    for (size_t i = 0; i < mCurrentFrame->N; i++) {
        float z = mCurrentFrame->mvKeys3Dc[i].z;
        if (z > 0)
            vDepthIdx.push_back(make_pair(z, i));
    }

    if (!vDepthIdx.empty()) {
        sort(vDepthIdx.begin(), vDepthIdx.end());

        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            size_t i = vDepthIdx[j].second;

            bool bCreateNew = false;

            Landmark* pMP = mCurrentFrame->GetLandmark(i);
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                bCreateNew = true;
                mCurrentFrame->AddLandmark(static_cast<Landmark*>(nullptr), i);
            }

            if (bCreateNew) {
                cv::Mat x3D = mCurrentFrame->UnprojectWorld(i);
                Landmark* pNewLM = new Landmark(x3D, pKF, mpMap);
                pNewLM->AddObservation(pKF, i);
                pKF->AddLandmark(pNewLM, i);
                pNewLM->ComputeDistinctiveDescriptors();
                mpMap->AddLandmark(pNewLM);

                mCurrentFrame->AddLandmark(pNewLM, i);
                nPoints++;
            } else {
                nPoints++;
            }

            if (vDepthIdx[j].first > 3.09f && nPoints > 100)
                break;
        }
    }

    mnLastKFid = mCurrentFrame->GetId();
    mpLastKF = pKF;
}
