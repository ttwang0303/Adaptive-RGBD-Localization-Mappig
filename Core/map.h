#ifndef MAP_H
#define MAP_H

#include <mutex>
#include <set>
#include <vector>

class Frame;
class Landmark;

class Map {
public:
    Map();

    void AddKeyFrame(Frame* pKF);
    void AddLandmark(Landmark* pLMK);

    std::vector<Frame*> GetAllKeyFrames();
    std::vector<Landmark*> GetAllLandmarks();

    long unsigned int LandmarksInMap();
    long unsigned KeyFramesInMap();

    void Clear();

    std::mutex mMutexMapUpdate;

protected:
    std::set<Landmark*> mspLandmarks;
    std::set<Frame*> mspKeyFrames;

    std::mutex mMutexMap;
};

#endif // MAP_H
