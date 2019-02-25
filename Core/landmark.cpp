#include "landmark.h"
#include "Features/matcher.h"
#include "Utils/utils.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"

using namespace std;

long unsigned int Landmark::nNextId = 0;
mutex Landmark::mGlobalMutex;

Landmark::Landmark(const cv::Mat& Pos, KeyFrame* pKF, Map* pMap)
    : mnFirstKFid(pKF->mnId)
    , mnFirstFrame(pKF->mnFrameId)
    , nObs(0)
    , mnTrackReferenceForFrame(0)
    , mnLastFrameSeen(0)
    , mnBALocalForKF(0)
    , mnFuseCandidateForKF(0)
    , mpRefKF(pKF)
    , mnVisible(1)
    , mnFound(1)
    , mbBad(false)
    , mpReplaced(static_cast<Landmark*>(nullptr))
    , mpMap(pMap)
{
    Pos.copyTo(mWorldPos);

    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

Landmark::Landmark(const cv::Mat& Pos, Map* pMap, Frame* pFrame, const size_t& idx)
    : mnFirstKFid(-1)
    , mnFirstFrame(pFrame->mnId)
    , nObs(0)
    , mnTrackReferenceForFrame(0)
    , mnLastFrameSeen(0)
    , mnBALocalForKF(0)
    , mnFuseCandidateForKF(0)
    , mpRefKF(static_cast<KeyFrame*>(nullptr))
    , mnVisible(1)
    , mnFound(1)
    , mbBad(false)
    , mpReplaced(nullptr)
    , mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    pFrame->mDescriptors.row(idx).copyTo(mDescriptor);

    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
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

void Landmark::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    nObs++;
}

void Landmark::EraseObservation(KeyFrame* pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF)) {
            int idx = mObservations[pKF];
            nObs--;
            mObservations.erase(pKF);

            if (mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // If only 2 observations or less, discad point
            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad)
        SetBadFlag();
}

void Landmark::SetBadFlag()
{
    map<KeyFrame*, size_t> observations;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        observations = mObservations;
        mObservations.clear();
    }

    for (auto& [pKF, idx] : observations)
        pKF->EraseLandmark(idx);

    mpMap->EraseLandmark(this);
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

int Landmark::GetIndexInKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool Landmark::IsInKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

bool Landmark::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void Landmark::Replace(Landmark* pLM)
{
    if (pLM->mnId == this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad = true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pLM;
    }

    for (auto& [pKF, idx] : obs) {
        if (!pLM->IsInKeyFrame(pKF)) {
            pKF->ReplaceLandmark(idx, pLM);
            pLM->AddObservation(pKF, idx);
        } else {
            pKF->EraseLandmark(idx);
        }
    }

    pLM->IncreaseFound(nfound);
    pLM->IncreaseVisible(nvisible);
    pLM->ComputeDistinctiveDescriptors();

    mpMap->EraseLandmark(this);
}

Landmark* Landmark::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void Landmark::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

void Landmark::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}

float Landmark::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}

void Landmark::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;
    map<KeyFrame*, size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if (mbBad)
            return;
        observations = mObservations;
    }
    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());
    for (auto& [pKF, idx] : observations) {
        if (!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(idx));
    }
    if (vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();
    double Distances[N][N];

    for (size_t i = 0; i < N; i++) {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++) {
            double distij = Matcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    double BestMedian = numeric_limits<double>::max();
    size_t BestIdx = 0;
    for (size_t i = 0; i < N; i++) {
        vector<double> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());
        double median = vDists[0.5 * (N - 1)];

        if (median < BestMedian) {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat Landmark::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}
