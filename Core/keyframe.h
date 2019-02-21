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

    std::vector<size_t> GetFeaturesInArea(const float& x, const float& y, const float& r) const;

    bool IsInImage(const float& x, const float& y) const;

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) { return pKF1->mnId < pKF2->mnId; }

public:
    static long unsigned int nNextKFid;
    const long unsigned int mnFrameId;

    long unsigned int mnFuseTargetForKF;

    // Used by Database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;

    // Manage covisibility graph, spanning tree and loop edges
    CovisibilityGraph mG;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

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
