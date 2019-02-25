#include "localmapping.h"
#include "Core/keyframe.h"
#include "Core/landmark.h"
#include "Core/map.h"
#include "Features/matcher.h"
#include "Odometry/localbundleadjustment.h"
#include <iostream>

using namespace std;

LocalMapping::LocalMapping(Map* pMap, DBoW3::Vocabulary* pVoc)
    : mbResetRequested(false)
    , mbFinishRequested(false)
    , mbFinished(true)
    , mpVocabulary(pVoc)
    , mpMap(pMap)
    , mbAbortBA(false)
    , mbStopped(false)
    , mbStopRequested(false)
    , mbNotStop(false)
    , mbAcceptKFs(true)
{
}

void LocalMapping::Run()
{
    mbFinished = false;

    while (true) {
        SetAcceptKFs(false);

        if (CheckNewKeyFrames()) {
            ProcessNewKeyFrame();

            LandmarkCulling();

            if (!CheckNewKeyFrames())
                FuseLandmarks();

            mbAbortBA = false;

            if (!CheckNewKeyFrames() && !StopRequested()) {
                if (mpMap->KeyFramesInMap() > 2)
                    LocalBundleAdjustment::Compute(mpCurrentKF, &mbAbortBA, mpMap);

                KeyFrameCulling();
            }
        } else if (Stop()) {
            // Safe area to stop
            while (isStopped() && !CheckFinish())
                usleep(3000);

            if (CheckFinish())
                break;
        }

        ResetIfRequested();
        SetAcceptKFs(true);

        if (CheckFinish())
            break;
        usleep(3000);
    }

    SetFinish();
}

bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mKeyFramesQueue.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKF = mKeyFramesQueue.front();
        mKeyFramesQueue.pop_front();
    }

    mpCurrentKF->ComputeBoW(mpVocabulary);

    // Associate Landmarks to the new KF
    const vector<Landmark*> vpLandmarks = mpCurrentKF->GetLandmarks();
    for (size_t i = 0; i < vpLandmarks.size(); i++) {
        Landmark* pLM = vpLandmarks[i];
        if (!pLM)
            continue;
        if (pLM->isBad())
            continue;

        if (!pLM->IsInKeyFrame(mpCurrentKF)) {
            pLM->AddObservation(mpCurrentKF, i);
            pLM->ComputeDistinctiveDescriptors();
        }
        // This only happen for new points inserted
        else {
            mlpRecentLandmarks.push_back(pLM);
        }
    }

    mpCurrentKF->UpdateConnections();
    mpMap->AddKeyFrame(mpCurrentKF);
}

void LocalMapping::LandmarkCulling()
{
    list<Landmark*>::iterator it = mlpRecentLandmarks.begin();
    const unsigned long int nCurrentKFid = mpCurrentKF->mnId;

    int nThObs = 2;
    const int cnThObs = nThObs;

    while (it != mlpRecentLandmarks.end()) {
        Landmark* pLM = *it;
        if (pLM->isBad()) {
            it = mlpRecentLandmarks.erase(it);
        } else if (pLM->GetFoundRatio() < 0.25f) {
            pLM->SetBadFlag();
            it = mlpRecentLandmarks.erase(it);
        } else if (((int)nCurrentKFid - (int)pLM->mnFirstKFid) >= 2 && pLM->Observations() <= cnThObs) {
            pLM->SetBadFlag();
            it = mlpRecentLandmarks.erase(it);
        } else if (((int)nCurrentKFid - (int)pLM->mnFirstKFid) >= 3) {
            it = mlpRecentLandmarks.erase(it);
        } else {
            it++;
        }
    }
}

void LocalMapping::FuseLandmarks()
{
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(10);
    vector<KeyFrame*> vpTargetKFs;

    for (KeyFrame* pKFi : vpNeighKFs) {
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKF->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKF->mnId;

        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (KeyFrame* pKFi2 : vpSecondNeighKFs) {
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKF->mnId || pKFi2->mnId == mpCurrentKF->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    Matcher matcher;
    vector<Landmark*> vpLandmarks = mpCurrentKF->GetLandmarks();
    for (KeyFrame* pKFi : vpTargetKFs)
        matcher.Fuse(pKFi, vpLandmarks, 3);

    // Search matches by projection from target KFs in current KF
    vector<Landmark*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpLandmarks.size());
    for (KeyFrame* pKFi : vpTargetKFs) {
        vector<Landmark*> vpLandmarksKFi = pKFi->GetLandmarks();

        for (Landmark* pLMi : vpLandmarksKFi) {
            if (!pLMi)
                continue;
            if (pLMi->isBad() || pLMi->mnFuseCandidateForKF == mpCurrentKF->mnId)
                continue;

            pLMi->mnFuseCandidateForKF = mpCurrentKF->mnId;
            vpFuseCandidates.push_back(pLMi);
        }
    }

    matcher.Fuse(mpCurrentKF, vpFuseCandidates, 3);

    // Update points
    vpLandmarks = mpCurrentKF->GetLandmarks();
    for (Landmark* pLM : vpLandmarks) {
        if (pLM) {
            if (!pLM->isBad()) {
                pLM->ComputeDistinctiveDescriptors();
            }
        }
    }

    mpCurrentKF->UpdateConnections();
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant KFs (only local KFs)
    // A KF is considered redundant if the 90% of the Landmarks it sees, are seen in at least
    // other 3 KFs, only consider close points
    vector<KeyFrame*> vpLocalKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    for (KeyFrame* pKF : vpLocalKFs) {
        if (pKF->mnId == 0)
            continue;

        vector<Landmark*> vpLandmarks = pKF->GetLandmarks();

        const int thObs = 3;
        int nRedundantObs = 0;
        int nLMs = 0;
        for (size_t i = 0; i < vpLandmarks.size(); ++i) {
            Landmark* pLM = vpLandmarks[i];
            if (!pLM)
                continue;
            if (pLM->isBad())
                continue;
            if (pKF->mvKeys3Dc[i].z > 3.09f || pKF->mvKeys3Dc[i].z < 0)
                continue;

            nLMs++;
            if (pLM->Observations() > thObs) {
                const map<KeyFrame*, size_t> observations = pLM->GetObservations();
                int nObs = 0;
                for (auto& [pKFi, idx] : observations) {
                    if (pKFi == pKF)
                        continue;
                    nObs++;
                    if (nObs > thObs)
                        break;
                }
                if (nObs >= thObs)
                    nRedundantObs++;
            }
        }

        if (nRedundantObs > 0.99 * nLMs)
            pKF->SetBadFlag();
    }
}

void LocalMapping::InsertKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mKeyFramesQueue.push_back(pKF);
    mbAbortBA = true;
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop) {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::StopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if (mbFinished)
        return;

    mbStopped = false;
    mbStopRequested = false;

    for (list<KeyFrame*>::iterator lit = mKeyFramesQueue.begin(); lit != mKeyFramesQueue.end(); lit++)
        delete *lit;

    mKeyFramesQueue.clear();
    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKFs()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKFs;
}

void LocalMapping::SetAcceptKFs(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKFs = flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if (flag && mbStopped)
        return false;

    mbNotStop = flag;
    return true;
}

void LocalMapping::InterruptBA() { mbAbortBA = true; }

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (1) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested) {
        mKeyFramesQueue.clear();
        mlpRecentLandmarks.clear();
        mbResetRequested = false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}
