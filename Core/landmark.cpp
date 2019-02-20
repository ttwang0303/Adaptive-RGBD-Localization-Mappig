#include "landmark.h"
#include "Utils/utils.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"

using namespace std;

long unsigned int Landmark::nNextId = 0;
mutex Landmark::mGlobalMutex;

Landmark::Landmark(const cv::Mat& Pos, Map* pMap, KeyFrame* pKF, const size_t& idx)
    : mnFirstKFid(pKF->GetId())
    , mnFirstFrame(pKF->mnFrameId)
    , nObs(0)
    , mnLastFrameSeen(0)
    , mbBad(false)
    , mpMap(pMap)
{
    Pos.copyTo(mWorldPos);

    pKF->mDescriptors.row(idx).copyTo(mDescriptor);

    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

Landmark::Landmark(const cv::Mat& Pos, Map* pMap, Frame& pF, const size_t& idx)
    : mnFirstKFid(-1)
    , mnFirstFrame(pF.GetId())
    , nObs(0)
    , mnLastFrameSeen(0)
    , mbBad(false)
    , mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    pF.mDescriptors.row(idx).copyTo(mDescriptor);

    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

void Landmark::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    nObs++;
}

map<KeyFrame*, size_t> Landmark::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int Landmark::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void Landmark::Covisibility(map<KeyFrame*, int>& KFcounter, unsigned long noId)
{
    unique_lock<mutex> lock(mMutexFeatures);
    for (auto& [pKFobs, idx] : mObservations) {
        if (pKFobs->GetId() == noId)
            continue;
        KFcounter[pKFobs]++;
    }
}

bool Landmark::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void Landmark::SetWorldPos(const cv::Mat& Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat Landmark::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}
