#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "DBoW3/DBoW3.h"
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <mutex>

class Map;
class KeyFrame;
class Database;
class LocalMapping;

class LoopClosing {
public:
public:
    typedef std::pair<std::set<KeyFrame*>, int> ConsistentGroup;

    typedef std::map<
        KeyFrame*, g2o::Sim3,
        std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3>>>
        KeyFrameAndPose;

public:
    LoopClosing(Map* pMap, Database* pDB, DBoW3::Vocabulary* pVoc);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    void RequestReset();
    void RequestFinish();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    Database* mpDB;
    DBoW3::Vocabulary* mpVocabulary;

    LocalMapping* mpLocalMapper;

    std::list<KeyFrame*> mlpKeyFrameQueue;
    std::mutex mMutexQueue;

    // Loop detector variables
    KeyFrame* mpCurrentKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;

    long unsigned int mLastLoopKFid;
};

#endif // LOOPCLOSING_H
