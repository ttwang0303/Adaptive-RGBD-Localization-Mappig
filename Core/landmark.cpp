#include "landmark.h"
#include "Utils/utils.h"
#include "keyframe.h"

using namespace std;

long unsigned int Landmark::nNextId = 0;
mutex Landmark::mGlobalMutex;

Landmark::Landmark(const cv::Mat& Pos, KeyFrame* pKF, const int& idx)
    : mnFirstFrame(pKF->GetId())
    , nObs(0)
{
    Pos.copyTo(mWorldPos);
    mnId = nNextId++;

    pKF->mDescriptors.row(idx).copyTo(mDescriptor);
}

void Landmark::AddObservation(KeyFrame* pKF, size_t idx)
{
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    nObs++;
}

int Landmark::Observations()
{
    return nObs;
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
