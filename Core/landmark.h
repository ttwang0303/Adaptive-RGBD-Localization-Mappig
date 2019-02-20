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
    Landmark(const cv::Mat& Pos, Map* pMap, KeyFrame* pKF, const size_t& idx);

    Landmark(const cv::Mat& Pos, Map* pMap, Frame& pF, const size_t& idx);

    void SetWorldPos(const cv::Mat& Pos);
    cv::Mat GetWorldPos();

    std::map<KeyFrame*, size_t> GetObservations();
    void AddObservation(KeyFrame* pKF, size_t idx);
    int Observations();

    // Check in which KFs this landmark is seen. Increase counter for those KFs
    // Exclude noId
    void Covisibility(std::map<KeyFrame*, int>& KFcounter, long unsigned int noId);

    bool isBad();

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Used by Tracking
    bool mbTrackInView;
    long unsigned int mnLastFrameSeen;

    static std::mutex mGlobalMutex;

private:
    // Position in absolute coordinates
    cv::Mat mWorldPos;

    // Frames observing the point and associated index in Frame
    std::map<KeyFrame*, size_t> mObservations;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Bad flag
    bool mbBad;

    Map* mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

#endif // LANDMARK_H
