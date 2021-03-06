#ifndef TRACKING_H
#define TRACKING_H

#include "Core/frame.h"
#include "Utils/common.h"
#include <DBoW3/Vocabulary.h>

class KeyFrame;
class Map;
class Landmark;
class Database;
class Extractor;
class Odometry;
class LocalMapping;
class Viewer;

class Tracking {
public:
    Tracking(DBoW3::Vocabulary* pVoc, Map* pMap, Database* pKFDB, Extractor* pExtractor, Viewer* pViewer);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    void SetOdometer(Odometry* pOdometer);

    cv::Mat Track(const cv::Mat& imColor, const cv::Mat& imDepth, const double& timestamp);

    void SaveTrajectory(const std::string& filename);
    void SaveKeyFrameTrajectory(const std::string& filename);
    void SaveObservationHistogram(const std::string& filename);

public:
    enum eTrackingState {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Current Frame
    Ptr<Frame> mpCurrentFrame;

    std::list<cv::Mat> mlRelativeFramePoses;
    std::list<KeyFrame*> mlpReferences;
    std::list<double> mlFrameTimes;

protected:
    void Initialization();

    void CheckReplaced();
    bool TrackFrame();

    void UpdateLastFrame();
    int DiscardOutliers();
    void CleanVOmatches();
    void DeleteTemporalPoints();

    double tNorm(const cv::Mat& T);
    double RNorm(const cv::Mat& T);
    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Local map functions
    bool TrackLocalMap();
    void UpdateLocalKFs();
    void UpdateLocalLMs();
    void SearchLocalLMs();

    void UpdateRelativePose();

    LocalMapping* mpLocalMapper;

    // BoW
    DBoW3::Vocabulary* mpVocabulary;
    Database* mpKeyFrameDB;

    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKFs;
    std::vector<Landmark*> mvpLocalLMs;

    Map* mpMap;

    Viewer* mpViewer;

    Extractor* mpExtractor;

    Odometry* mpOdometer;

    KeyFrame* mpLastKF;
    Ptr<Frame> mpLastFrame;
    unsigned int mnLastKFid;

    std::list<Landmark*> mlpTemporalPoints;
};

#endif // TRACKING_H
