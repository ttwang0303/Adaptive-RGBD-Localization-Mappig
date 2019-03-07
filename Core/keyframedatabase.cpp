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
    set<KeyFrame*> spConnectedKFs = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;

    // Search KeyFrames that share a word with current KeyFrame
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(); vit != pKF->mBowVec.end(); vit++) {
            const DBoW3::WordId& wordId = vit->first;
            list<KeyFrame*> lKFs = mvInvertedFile[wordId];

            for (KeyFrame* pKFsw : lKFs) {
                if (pKFsw->mnLoopQuery != pKF->mnId) {
                    pKFsw->mnLoopWords = 0;
                    if (!spConnectedKFs.count(pKFsw)) {
                        pKFsw->mnLoopQuery = pKF->mnId;
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

    list<pair<float, KeyFrame*>> lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Accumulate score by covisibility
    for (auto& pair : lScoreAndMatch) {
        KeyFrame* pKFi = pair.second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = pair.first;
        float accScore = pair.first;
        KeyFrame* pBestKF = pKFi;
        for (KeyFrame* pKF2 : vpNeighs) {
            if (pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords) {
                accScore += pKF2->mLoopScore;
                if (pKF2->mLoopScore > bestScore) {
                    pBestKF = pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back({ accScore, pBestKF });
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return all KFs with score > 0.75*bestScore
    float minScoreToRetain = 0.75f * bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (auto& pair : lAccScoreAndMatch) {
        float score = pair.first;
        if (score > minScoreToRetain) {
            KeyFrame* pKFi = pair.second;
            if (!spAlreadyAddedKF.count(pKFi)) {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}
