#ifndef LANDMARK_H
#define LANDMARK_H

#include <map>
#include <mutex>
#include <opencv2/core.hpp>

class KeyFrame;
class Map;
class Frame;

class Landmark {
public:
    Landmark(const cv::Mat& Pos, KeyFrame* pKF, Map* pMap);
    Landmark(const cv::Mat& Pos, Map* pMap, Frame* pFrame, const size_t& idx);

    void SetWorldPos(const cv::Mat& Pos);
    cv::Mat GetWorldPos();

    std::map<KeyFrame*, size_t> GetObservations();
    void AddObservation(KeyFrame* pKF, size_t idx);
    void EraseObservation(KeyFrame* pKF);
    int Observations();

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(Landmark* pLM);
    Landmark* GetReplaced();

    void IncreaseVisible(int n = 1);
    void IncreaseFound(int n = 1);
    float GetFoundRatio();

    void ComputeDistinctiveDescriptors();
    cv::Mat GetDescriptor();

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Used by Tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    // Used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

private:
    // Position in absolute coordinates
    cv::Mat mWorldPos;

    // Frames observing the point and associated index in Frame
    std::map<KeyFrame*, size_t> mObservations;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    KeyFrame* mpRefKF;

    int mnVisible;
    int mnFound;

    // Bad flag
    bool mbBad;
    Landmark* mpReplaced;

    Map* mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

#endif // LANDMARK_H
