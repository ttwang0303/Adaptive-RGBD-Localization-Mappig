#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include <DBoW3/Vocabulary.h>
#include <list>
#include <mutex>

class Map;
class Landmark;
class KeyFrame;

class LocalMapping {
public:
    LocalMapping(Map* pMap, DBoW3::Vocabulary* pVoc);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool StopRequested();
    bool AcceptKFs();
    void SetAcceptKFs(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyFramesInQueue()
    {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        return mKeyFramesQueue.size();
    }

protected:
    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void LandmarkCulling();
    void FuseLandmarks();
    void KeyFrameCulling();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    DBoW3::Vocabulary* mpVocabulary;

    Map* mpMap;

    std::list<KeyFrame*> mKeyFramesQueue;

    KeyFrame* mpCurrentKF;

    std::list<Landmark*> mlpRecentLandmarks;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKFs;
    std::mutex mMutexAccept;
};

#endif // LOCALMAPPING_H
