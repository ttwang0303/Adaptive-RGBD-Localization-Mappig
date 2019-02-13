#ifndef LANDMARK_H
#define LANDMARK_H

#include <map>
#include <mutex>
#include <opencv2/core.hpp>

class Frame;

class Landmark {
public:
    Landmark(const cv::Mat& Pos, Frame* pFrame, const int& idxF);

    void SetWorldPos(const cv::Mat& Pos);
    cv::Mat GetWorldPos();

    void AddObservation(Frame* pFrame, size_t idx);
    int Observations();

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstFrame;
    int nObs;

    static std::mutex mGlobalMutex;

private:
    // Position in absolute coordinates
    cv::Mat mWorldPos;

    // Frames observing the point and associated index in Frame
    std::map<Frame*, size_t> mObservations;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

#endif // LANDMARK_H
