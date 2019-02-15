#include "keyframedatabase.h"
#include "keyframe.h"
#include <cmath>

using namespace std;

Database::Database(DBoW3::Vocabulary* pVoc)
    : mpVoc(pVoc)
{
    mvInvertedFile.resize(pVoc->size());
}

void Database::Add(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(); vit != pKF->mBowVec.end(); vit++) {
        const DBoW3::WordId& wordId = vit->first;
        mvInvertedFile[wordId].push_back(pKF);
    }
}

void Database::Erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(); vit != pKF->mBowVec.end(); vit++) {
        // List of keyframes that share the word
        const DBoW3::WordId& wordId = vit->first;
        list<KeyFrame*>& lKFs = mvInvertedFile[wordId];

        for (list<KeyFrame*>::iterator lit = lKFs.begin(); lit != lKFs.end(); lit++) {
            if (pKF == *lit) {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void Database::Clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

vector<KeyFrame*> Database::Query(KeyFrame* pKF, float minScore)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search KeyFrames that share a word with current KeyFrame
    // Simulate connected KeyFrames with distance of 10
    {
        unique_lock<mutex> lock(mMutex);

        for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(); vit != pKF->mBowVec.end(); vit++) {
            const DBoW3::WordId& wordId = vit->first;
            list<KeyFrame*> lKFs = mvInvertedFile[wordId];

            for (KeyFrame* pKFsw : lKFs) {
                if (pKFsw->mnLoopQuery != pKF->GetId()) {
                    pKFsw->mnLoopWords = 0;
                    if (abs(static_cast<int>(pKF->GetId() - pKFsw->GetId())) > 5) {
                        pKFsw->mnLoopQuery = pKF->GetId();
                        lKFsSharingWords.push_back(pKFsw);
                    }
                }
                pKFsw->mnLoopWords++;
            }
        }
    }

    if (lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*>> lScoreAndMatch;

    // Only compare against those KeyFrames that share enough words
    int maxCommonWords = 0;
    for (KeyFrame* pKFsw : lKFsSharingWords) {
        if (pKFsw->mnLoopWords > maxCommonWords)
            maxCommonWords = pKFsw->mnLoopWords;
    }

    int minCommonWords = maxCommonWords * 0.8f;

    // Compute similarity score. Retain the matches whose score is higher that minScore
    for (KeyFrame* pKFsw : lKFsSharingWords) {
        if (pKFsw->mnLoopWords > minCommonWords) {
            float score = mpVoc->score(pKF->mBowVec, pKFsw->mBowVec);
            pKFsw->mLoopScore = score;
            if (score >= minScore)
                lScoreAndMatch.push_back({ score, pKFsw });
        }
    }

    if (lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lScoreAndMatch.size());

    for (auto& pair : lScoreAndMatch) {
        KeyFrame* pKFi = pair.second;
        if (!spAlreadyAddedKF.count(pKFi)) {
            vpLoopCandidates.push_back(pKFi);
            spAlreadyAddedKF.insert(pKFi);
        }
    }

    return vpLoopCandidates;
}
