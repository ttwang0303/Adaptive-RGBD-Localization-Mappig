#include "map.h"
#include "keyframe.h"
#include "landmark.h"

using namespace std;

Map::Map()
{
}

void Map::AddKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
}

void Map::AddLandmark(Landmark* pLMK)
{
    unique_lock<mutex> lock(mMutexMap);
    mspLandmarks.insert(pLMK);
}

void Map::EraseLandmark(Landmark* pLM)
{
    unique_lock<mutex> lock(mMutexMap);
    mspLandmarks.erase(pLM);
}

void Map::EraseKF(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
}

vector<Landmark*> Map::GetAllLandmarks()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<Landmark*>(mspLandmarks.begin(), mspLandmarks.end());
}

unsigned long Map::LandmarksInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspLandmarks.size();
}

unsigned long Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

void Map::Clear()
{
    for (set<Landmark*>::iterator sit = mspLandmarks.begin(); sit != mspLandmarks.end(); sit++)
        delete *sit;

    for (set<KeyFrame*>::iterator sit = mspKeyFrames.begin(); sit != mspKeyFrames.end(); sit++)
        delete *sit;

    mspLandmarks.clear();
    mspKeyFrames.clear();
}
