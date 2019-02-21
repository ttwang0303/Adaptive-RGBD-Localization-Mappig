#include "keyframe.h"
#include "covisiblegraph.h"
#include "keyframedatabase.h"
#include "landmark.h"
#include "map.h"

using namespace std;

long unsigned int KeyFrame::nNextKFid = 0;

KeyFrame::KeyFrame(Frame& frame, Map* pMap, Database* pKFDB)
    : mnFrameId(frame.GetId())
    , mnFuseTargetForKF(0)
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

    mvpLandmarks = frame.GetLandmarks();
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

vector<size_t> KeyFrame::GetFeaturesInArea(const float& x, const float& y, const float& r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        const cv::KeyPoint& kp = mvKeys[i];

        const float distx = kp.pt.x - x;
        const float disty = kp.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r)
            vIndices.push_back(i);
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float& x, const float& y) const
{
    return (x >= 0.0f && x < mImGray.cols && y >= 0.0f && y < mImGray.rows);
}

void KeyFrame::SetBadFlag()
{
    if (GetId() == 0)
        return;
    else if (mbNotErase) {
        mbToBeErased = true;
        return;
    }

    mG.EraseRootConnections();

    {
        unique_lock<mutex> lockFeat(mMutexFeatures);
        for (size_t i = 0; i < mvpLandmarks.size(); ++i) {
            if (mvpLandmarks[i])
                mvpLandmarks[i]->EraseObservation(this);
        }
    }

    mG.UpdateSpanningTree();
    mTcw = GetPose() * mG.GetParent()->GetPoseInv();
    mbBad = true;

    mpMap->EraseKF(this);
    mpKeyFrameDB->Erase(this);
}

bool KeyFrame::isBad()
{
    return mbBad;
}
