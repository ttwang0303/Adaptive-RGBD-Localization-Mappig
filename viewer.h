#ifndef VIEWER_H
#define VIEWER_H

#include <mutex>

class PointCloudDrawer;

class Viewer {
public:
    Viewer(PointCloudDrawer* pCloudDrawer);

    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    void Release();

private:
    bool Stop();

    PointCloudDrawer* mpCloudDrawer;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;
};

#endif // VIEWER_H
