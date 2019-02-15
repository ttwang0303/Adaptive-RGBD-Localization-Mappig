#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "frame.h"

class Map;
class Database;

class KeyFrame : public Frame {
public:
    KeyFrame(Frame& frame, Map* pMap, Database* pKFDB);

    static bool weightComp(int a, int b) { return a > b; }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) { return pKF1->mnId < pKF2->mnId; }

public:
    static long unsigned int nNextKFid;
    const long unsigned int mnFrameId;

protected:
    // BoW
    Database* mpKeyFrameDB;

    Map* mpMap;
};

#endif // KEYFRAME_H
