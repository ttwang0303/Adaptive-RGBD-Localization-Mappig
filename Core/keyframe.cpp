#include "keyframe.h"
#include "keyframedatabase.h"
#include "landmark.h"
#include "map.h"

using namespace std;

long unsigned int KeyFrame::nNextKFid = 0;

KeyFrame::KeyFrame(Frame& frame, Map* pMap, Database* pKFDB)
    : mnFrameId(frame.mnId)
    , mnTrackReferenceForFrame(0)
    , mnFuseTargetForKF(0)
    , mnBALocalForKF(0)
    , mnBAFixedForKF(0)
    , mnLoopQuery(0)
    , mnLoopWords(0)
    , mpKeyFrameDB(pKFDB)
    , mbFirstConnection(true)
    , mpParent(nullptr)
    , mbNotErase(false)
    , mbToBeErased(false)
    , mbBad(false)
    , mpMap(pMap)
{
    mImColor = frame.mImColor;
    mImGray = frame.mImGray;
    mImDepth = frame.mImDepth;
    mTimestamp = frame.mTimestamp;
    mvKeys = frame.mvKeys;
    mvKeysUn = frame.mvKeysUn;
    mvKeys3Dc = frame.mvKeys3Dc;
    mvuRight = frame.mvuRight;
    mnMinX = frame.mnMinX;
    mnMinY = frame.mnMinY;
    mnMaxX = frame.mnMaxX;
    mnMaxY = frame.mnMaxY;
    mK = frame.mK;
    mDistCoef = frame.mDistCoef;
    mDescriptors = frame.mDescriptors;
    mBowVec = frame.mBowVec;
    mFeatVec = frame.mFeatVec;
    N = frame.N;
    mvbOutlier = frame.GetOutliers();

    if (frame.mpCloud)
        mpCloud = frame.mpCloud;

    cv::Mat framePose = frame.GetPose();
    if (!framePose.empty())
        SetPose(framePose);

    mvpLandmarks = frame.GetLandmarks();

    mnId = nNextKFid++;
}

void KeyFrame::SetPose(cv::Mat Tcw)
{
    unique_lock<mutex> lock(mMutexPose);
    Frame::SetPose(Tcw);
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw.clone();
}

cv::Mat KeyFrame::GetPoseInv()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetRotationInv()
{
    unique_lock<mutex> lock(mMutexPose);
    return mRwc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return mOw.clone();
}

cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mRcw.clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mtcw.clone();
}

void KeyFrame::AddConnection(KeyFrame* pKF, const int& weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF] = weight;
        else if (mConnectedKeyFrameWeights[pKF] != weight)
            mConnectedKeyFrameWeights[pKF] = weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mConnectedKeyFrameWeights.count(pKF)) {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate = true;
        }
    }

    if (bUpdate)
        UpdateBestCovisibles();
}

void KeyFrame::UpdateConnections()
{
    map<KeyFrame*, int> KFcounter;
    vector<Landmark*> vpLMs;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpLMs = mvpLandmarks;
    }

    // For all map points in keyframe check in which other keyframes are they seen
    // Increase counter for those keyframes
    for (Landmark* pLM : vpLMs) {
        if (!pLM)
            continue;
        if (pLM->isBad())
            continue;

        map<KeyFrame*, size_t> observations = pLM->GetObservations();

        for (auto& [pKF, idx] : observations) {
            if (pKF->mnId == mnId)
                continue;
            KFcounter[pKF]++;
        }
    }

    // This should not happen
    if (KFcounter.empty())
        return;

    // If the counter is greater than threshold add connection
    // In case no keyframe counter is over threshold add the one with maximum counter
    int nmax = 0;
    KeyFrame* pKFmax = nullptr;
    int th = 15;

    vector<pair<int, KeyFrame*>> vPairs;
    vPairs.reserve(KFcounter.size());
    for (auto& [pKF, n] : KFcounter) {
        if (n > nmax) {
            nmax = n;
            pKFmax = pKF;
        }
        if (n >= th) {
            vPairs.push_back({ n, pKF });
            pKF->AddConnection(this, n);
        }
    }

    if (vPairs.empty()) {
        vPairs.push_back({ nmax, pKFmax });
        pKFmax->AddConnection(this, nmax);
    }

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for (auto& [w, pKF] : vPairs) {
        lKFs.push_front(pKF);
        lWs.push_front(w);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if (mbFirstConnection && mnId != 0) {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }
    }
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int, KeyFrame*>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for (auto& [pKF, w] : mConnectedKeyFrameWeights)
        vPairs.push_back({ w, pKF });

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for (auto& [w, pKF] : vPairs) {
        lKFs.push_front(pKF);
        lWs.push_front(w);
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for (auto& [pKF, w] : mConnectedKeyFrameWeights)
        s.insert(pKF);

    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int& N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int& w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, KeyFrame::weightComp);
    if (it == mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else {
        int n = it - mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
    }
}

int KeyFrame::GetWeight(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLandmark(Landmark* pLandmark, const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpLandmarks[idx] = pLandmark;
}

vector<Landmark*> KeyFrame::GetLandmarks()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpLandmarks;
}

Landmark* KeyFrame::GetLandmark(const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpLandmarks[idx];
}

void KeyFrame::ReleaseLandmark(const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpLandmarks[idx] = static_cast<Landmark*>(nullptr);
}

void KeyFrame::ReplaceLandmark(const size_t& idx, Landmark* pLM)
{
    mvpLandmarks[idx] = pLM;
}

set<Landmark*> KeyFrame::GetLandmarkSet()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<Landmark*> s;
    for (Landmark* pLMK : mvpLandmarks) {
        if (!pLMK)
            continue;

        s.insert(pLMK);
    }
    return s;
}

void KeyFrame::EraseLandmark(const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpLandmarks[idx] = static_cast<Landmark*>(nullptr);
}

void KeyFrame::EraseLandmark(Landmark* pLM)
{
    int idx = pLM->GetIndexInKeyFrame(this);
    if (idx >= 0)
        mvpLandmarks[idx] = static_cast<Landmark*>(nullptr);
}

int KeyFrame::TrackedLandmarks(const int& minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints = 0;
    const bool bCheckObs = minObs > 0;
    for (size_t i = 0; i < N; i++) {
        Landmark* pLM = mvpLandmarks[i];
        if (pLM) {
            if (!pLM->isBad()) {
                if (bCheckObs) {
                    if (pLM->Observations() >= minObs)
                        nPoints++;
                } else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

cv::Mat KeyFrame::UnprojectWorld(const size_t& i)
{
    const cv::Point3f& p3Dc = mvKeys3Dc[i];
    if (p3Dc.z > 0) {
        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << p3Dc.x, p3Dc.y, p3Dc.z);

        unique_lock<mutex> lock(mMutexPose);
        return mRwc * x3Dc + mOw;
    } else
        return cv::Mat();
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        //        if (mspLoopEdges.empty()) {
        mbNotErase = false;
        //        }
    }

    if (mbToBeErased) {
        SetBadFlag();
    }
}

bool KeyFrame::IsInImage(const float& x, const float& y) const
{
    return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

void KeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mnId == 0)
            return;
        else if (mbNotErase) {
            mbToBeErased = true;
            return;
        }
    }

    for (auto& [pKF, w] : mConnectedKeyFrameWeights)
        pKF->EraseConnection(this);

    for (size_t i = 0; i < mvpLandmarks.size(); i++)
        if (mvpLandmarks[i])
            mvpLandmarks[i]->EraseObservation(this);

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with
        // highest covisibility weight) Include that children as new parent
        // candidate for the rest
        while (!mspChildrens.empty()) {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for (KeyFrame* pKF : mspChildrens) {
                if (pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0; i < vpConnected.size(); i++) {
                    for (KeyFrame* pKFpc : sParentCandidates) {
                        if (vpConnected[i]->mnId == pKFpc->mnId) {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if (w > max) {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if (bContinue) {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            } else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign
        // to the original parent of this KF
        if (!mspChildrens.empty()) {
            for (KeyFrame* pKFc : mspChildrens)
                pKFc->ChangeParent(mpParent);
        }

        mpParent->EraseChild(this);
        mTcp = mTcw * mpParent->GetPoseInv();
        mbBad = true;
    }

    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->Erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}
