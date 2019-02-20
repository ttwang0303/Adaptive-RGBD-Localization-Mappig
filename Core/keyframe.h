#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "covisiblegraph.h"
#include "frame.h"
#include <mutex>

class Map;
class Database;

class KeyFrame : public Frame {
public:
    KeyFrame(Frame& frame, Map* pMap, Database* pKFDB);

    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) { return pKF1->mnId < pKF2->mnId; }

public:
    static long unsigned int nNextKFid;
    const long unsigned int mnFrameId;

    // Used by Database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;

    // Manage covisibility graph, spanning tree and loop edges
    CovisibilityGraph mG;

protected:
    // BoW
    Database* mpKeyFrameDB;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    Map* mpMap;
};

#endif // KEYFRAME_H
