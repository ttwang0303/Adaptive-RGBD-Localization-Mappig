#include "viewer.h"
#include "Utils/common.h"
#include "mapdrawer.h"
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;

Viewer::Viewer(MapDrawer* pMapDrawer)
    : mpMapDrawer(pMapDrawer)
    , mbFinishRequested(false)
    , mbFinished(true)
    , mbStopped(true)
    , mbStopRequested(false)
    , mMeanTrackingTime(0)
    , mFusedLMs(0)
    , mMatched(0)
    , mInliers(0)
    , nKeypoints(0)
{
    mViewpointX = 0;
    mViewpointY = -0.7;
    mViewpointZ = -1.8;
    mViewpointF = 500;
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    pangolin::CreateWindowAndBind("Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<float> menuTrackTime("menu.Track time:", 0.0);
    pangolin::Var<size_t> menuKPs("menu.KPs:", 0);
    pangolin::Var<float> menuFusedLMs("menu.Fused LMs:", 0.0);
    pangolin::Var<int> menuMatches("menu.Matches:", 0.0);
    pangolin::Var<int> menuInliers("menu.Inliers:", 0.0);
    pangolin::Var<size_t> menuNodes("menu.Nodes:", 0);
    pangolin::Var<size_t> menuLandmarks("menu.Landmarks:", 0);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175),
                                    1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        menuLandmarks = mpMapDrawer->DrawLandmarks();
        menuNodes = mpMapDrawer->DrawKeyFrames(true, true);

        {
            unique_lock<mutex> locktime(mMutexUpdate);
            menuTrackTime = mMeanTrackingTime;
            menuFusedLMs = mFusedLMs;
            menuMatches = mMatched;
            menuInliers = mInliers;
            menuKPs = nKeypoints;
        }

        pangolin::FinishFrame();

        if (Stop()) {
            while (isStopped()) {
                usleep(3000);
            }
        }

        if (CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if (mbFinishRequested)
        return false;
    else if (mbStopRequested) {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;
}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

void Viewer::SetMeanTrackigTime(const float& meanTime)
{
    unique_lock<mutex> lock(mMutexUpdate);
    mMeanTrackingTime = meanTime;
}

void Viewer::SetFusedLMs(const int& nfused)
{
    unique_lock<mutex> lock(mMutexUpdate);
    mFusedLMs = nfused;
}

void Viewer::SetTotalMatches(const int n)
{
    unique_lock<mutex> lock(mMutexUpdate);
    mMatched = n;
}

void Viewer::SetTotalInliers(const int n)
{
    unique_lock<mutex> lock(mMutexUpdate);
    mInliers = n;
}

void Viewer::SetKPsDetected(const size_t& n)
{
    unique_lock<mutex> lock(mMutexUpdate);
    nKeypoints = n;
}
