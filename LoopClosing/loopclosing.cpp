#include "loopclosing.h"
#include "Core/keyframe.h"
#include "Core/keyframedatabase.h"
#include "Core/map.h"
#include "Features/matcher.h"
#include "Odometry/ransac.h"
#include "Utils/converter.h"

using namespace std;

LoopClosing::LoopClosing(Map* pMap, Database* pDB, DBoW3::Vocabulary* pVoc)
    : mbResetRequested(false)
    , mbFinishRequested(false)
    , mbFinished(true)
    , mpMap(pMap)
    , mpDB(pDB)
    , mpVocabulary(pVoc)
    , mLastLoopKFid(0)
{
}

void LoopClosing::Run()
{
    mbFinished = false;

    while (true) {
        if (CheckNewKFs()) {

            // Detect loop candidates
            if (DetectLoop()) {

                // Compute similarity transformation [R|t]
                if (ComputeSim3()) {
                }
            }
        }
    }
}

bool LoopClosing::CheckNewKFs()
{
    unique_lock<mutex> lock(mMutexKFsQueue);
    return (!mlpKFsQueue.empty());
}

bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexKFsQueue);
        mpCurrentKF = mlpKFsQueue.front();
        mlpKFsQueue.pop_front();
        // Avoid that a KF can be erased while its being process by this thread
        mpCurrentKF->SetNotErase();
    }

    // If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if (mpCurrentKF->GetId() < mLastLoopKFid + 10) {
        mpDB->Add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Query the database
    float minScore = 0.06f;
    vector<KeyFrame*> vpCandidateKFs = mpDB->Query(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new KF and return false
    if (vpCandidateKFs.empty()) {
        mpDB->Add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    mvpEnoughConsistentCandidates.clear();
    for (KeyFrame* pCandidateKF : vpCandidateKFs)
        mvpEnoughConsistentCandidates.push_back(pCandidateKF);

    mpDB->Add(mpCurrentKF);
    mpCurrentKF->SetErase();
    return true;
}

bool LoopClosing::ComputeSim3()
{
    const size_t nInitialCandidates = mvpEnoughConsistentCandidates.size();

    Matcher matcher(0.75f);

    vector<Ransac*> vpRansacSolvers;
    vpRansacSolvers.resize(nInitialCandidates);

    vector<vector<cv::DMatch>> vvMatches;
    vvMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates = 0;

    for (size_t i = 0; i < nInitialCandidates; i++) {
        KeyFrame* pKFc = mvpEnoughConsistentCandidates[i];
        pKFc->SetNotErase();

        int nmatches = matcher.BoWMatch(mpCurrentKF, pKFc, vvMatches[i]);

        if (nmatches < 20) {
            vbDiscarded[i] = true;
            continue;
        } else {
            Ransac* pSolver = new Ransac(mpCurrentKF, pKFc, vvMatches[i]);
            pSolver->SetParameters(200, 20, 3.0f, 4);
            vpRansacSolvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;
}
