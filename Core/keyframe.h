#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "frame.h"
#include <mutex>

class Map;
class Database;
class Landmark;

class KeyFrame : public Frame {
public:
    KeyFrame(Frame& frame, Map* pMap, Database* pKFDB);

    virtual ~KeyFrame() {}

    // Pose functions
    void SetPose(cv::Mat Tcw) override;
    cv::Mat GetPose() override;
    cv::Mat GetPoseInv() override;
    cv::Mat GetRotationInv() override;
    cv::Mat GetCameraCenter() override;
    cv::Mat GetRotation() override;
    cv::Mat GetTranslation() override;

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int& weight);
    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<KeyFrame*> GetConnectedKeyFrames();
    std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int& N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int& w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);

    // Landmark observation functions
    void AddLandmark(Landmark* pLandmark, const size_t& idx) override;
    std::vector<Landmark*> GetLandmarks() override;
    Landmark* GetLandmark(const size_t& idx) override;
    void ReleaseLandmark(const size_t& idx) override;
    void ReplaceLandmark(const size_t& idx, Landmark* pLM);
    std::set<Landmark*> GetLandmarkSet();
    void EraseLandmark(const size_t& idx);
    void EraseLandmark(Landmark* pLM);
    int TrackedLandmarks(const int& minObs);

    cv::Mat UnprojectWorld(const size_t& i) override;

    bool IsInImage(const float& x, const float& y) const;

    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    static bool weightComp(int a, int b) { return a > b; }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) { return pKF1->mnId < pKF2->mnId; }

public:
    static long unsigned int nNextKFid;
    const long unsigned int mnFrameId;

    // Used by Tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Used by Local Mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Used by Database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;

    cv::Mat mTcwGBA;
    long unsigned int mnBAGlobalForKF;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

protected:
    // BoW
    Database* mpKeyFrameDB;

    std::map<KeyFrame*, int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    Map* mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

#endif // KEYFRAME_H
