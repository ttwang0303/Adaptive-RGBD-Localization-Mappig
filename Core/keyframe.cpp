#include "keyframe.h"
#include "keyframedatabase.h"
#include "map.h"

using namespace std;

long unsigned int KeyFrame::nNextKFid = 0;

KeyFrame::KeyFrame(Frame& frame, Map* pMap, Database* pKFDB)
    : mnFrameId(frame.GetId())
    , mnLoopQuery(0)
    , mnLoopWords(0)
{
    mImColor = frame.mImColor;
    mImDepth = frame.mImDepth;
    mTimestamp = frame.mTimestamp;
    mvKps = frame.mvKps;
    mDescriptors = frame.mDescriptors;
    mvKps3Dc = frame.mvKps3Dc;
    mBowVec = frame.mBowVec;
    mFeatVec = frame.mFeatVec;
    N = frame.N;

    if (frame.mpCloud)
        mpCloud = frame.mpCloud;

    cv::Mat framePose = frame.GetPose();
    if (!framePose.empty())
        SetPose(framePose);

    mvpLandmarks = frame.GetLandmarksMatched();

    unique_lock<mutex> lock(mMutexId);
    mnId = nNextKFid++;
}
