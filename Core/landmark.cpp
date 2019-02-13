#include "landmark.h"
#include "Utils/utils.h"
#include "frame.h"

using namespace std;

long unsigned int Landmark::nNextId = 0;
mutex Landmark::mGlobalMutex;

Landmark::Landmark(const cv::Mat& Pos, Frame* pFrame, const int& idxF)
    : mnFirstFrame(pFrame->mnId)
    , nObs(0)
{
    Pos.copyTo(mWorldPos);
    mnId = nNextId++;

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);
}

void Landmark::AddObservation(Frame* pFrame, size_t idx)
{
    if (mObservations.count(pFrame))
        return;
    mObservations[pFrame] = idx;
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
