#ifndef LANDMARK_H
#define LANDMARK_H

#include <map>
#include <opencv2/core.hpp>

class Frame;

class Landmark {
public:
    Landmark(Frame* pFrame, const int& idxF);

    void AddObservation(Frame* pFrame, size_t idx);
    int Observations();

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstFrame;
    int nObs;

private:
    // Frames observing the point and associated index in Frame
    std::map<Frame*, size_t> mObservations;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;
};

#endif // LANDMARK_H
