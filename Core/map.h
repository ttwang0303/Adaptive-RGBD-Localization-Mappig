#ifndef MAP_H
#define MAP_H

#include <mutex>
#include <set>
#include <vector>

class KeyFrame;
class Landmark;

class Map {
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddLandmark(Landmark* pLMK);
    void EraseLandmark(Landmark* pLM);
    void EraseKF(KeyFrame* pKF);

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<Landmark*> GetAllLandmarks();

    long unsigned int LandmarksInMap();
    long unsigned KeyFramesInMap();

    void Clear();

    std::mutex mMutexMapUpdate;

    std::mutex mMutexPointCreation;

protected:
    std::set<Landmark*> mspLandmarks;
    std::set<KeyFrame*> mspKeyFrames;

    std::mutex mMutexMap;
};

#endif // MAP_H
