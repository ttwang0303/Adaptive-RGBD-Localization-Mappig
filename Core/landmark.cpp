#include "landmark.h"
#include "Utils/utils.h"
#include "frame.h"

using namespace std;

long unsigned int Landmark::nNextId = 0;

Landmark::Landmark(Frame* pFrame, const int& idxF)
    : mnFirstFrame(pFrame->mnId)
    , nObs(0)
{
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

