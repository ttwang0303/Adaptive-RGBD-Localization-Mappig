#ifndef VIEWER_H
#define VIEWER_H

#include <mutex>

class MapDrawer;

class Viewer {
public:
    Viewer(MapDrawer* pMapDrawer);

    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    void Release();

    void SetMeanTrackigTime(const float& meanTime);
    void SetFusedLMs(const int& nfused);
    void SetTotalMatches(const int n);
    void SetTotalInliers(const int n);
    void SetKPsDetected(const size_t& n);

private:
    bool Stop();

    MapDrawer* mpMapDrawer;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;

    std::mutex mMutexUpdate;
    float mMeanTrackingTime;
    float mFusedLMs;
    int mMatched;
    int mInliers;
    size_t nKeypoints;
};

#endif // VIEWER_H
