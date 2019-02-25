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
};

#endif // VIEWER_H
