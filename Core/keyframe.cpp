#include "keyframe.h"
#include "covisiblegraph.h"
#include "keyframedatabase.h"
#include "landmark.h"
#include "map.h"

using namespace std;

long unsigned int KeyFrame::nNextKFid = 0;

KeyFrame::KeyFrame(Frame& frame, Map* pMap, Database* pKFDB)
    : mnFrameId(frame.GetId())
    , mnLoopQuery(0)
    , mnLoopWords(0)
    , mpKeyFrameDB(pKFDB)
    , mbNotErase(false)
    , mbToBeErased(false)
    , mbBad(false)
    , mpMap(pMap)
{
    mImColor = frame.mImColor;
    mImGray = frame.mImGray;
    mImDepth = frame.mImDepth;
    mTimestamp = frame.mTimestamp;
    mvKeys = frame.mvKeys;
    mDescriptors = frame.mDescriptors;
    mvKeys3Dc = frame.mvKeys3Dc;
    mvbOutlier = frame.mvbOutlier;
    mBowVec = frame.mBowVec;
    mFeatVec = frame.mFeatVec;
    N = frame.N;

    if (frame.mpCloud)
        mpCloud = frame.mpCloud;

    cv::Mat framePose = frame.GetPose();
    if (!framePose.empty())
        SetPose(framePose);

    mvpLandmarks = frame.GetLandmarksMatched();
    mG.SetRootNode(this);

    unique_lock<mutex> lock(mMutexId);
    mnId = nNextKFid++;
}

void KeyFrame::SetNotErase()
{
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    if (mG.isLoopEmpty())
        mbNotErase = false;

    if (mbToBeErased) {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{

}
