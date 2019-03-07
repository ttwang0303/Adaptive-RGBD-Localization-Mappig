#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <opencv2/features2d.hpp>

class Extractor {
public:
    enum eAlgorithm {
        ORB = 0,
        ORB_SLAM2,
        FAST,
        GFTT,
        STAR,
        BRISK,
        FREAK,
        BRIEF,
        LATCH,
        SURF,
        SIFT
    };

    enum eMode {
        NORMAL = 0,
        ADAPTIVE
    };

    eAlgorithm mDetectorAlgorithm;
    eAlgorithm mDescriptorAlgorithm;
    eMode mMode;

    Extractor(const eAlgorithm& detector = SURF, const eAlgorithm& descriptor = BRIEF, const eMode& mode = ADAPTIVE);

    void Extract(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);

    static int mNorm;

private:
    void CreateAdaptiveDetector();
    void CreateDetector();
    void CreateDescriptor();

    cv::Ptr<cv::FeatureDetector> mpDetector;
    cv::Ptr<cv::DescriptorExtractor> mpDescriptor;
};

#endif // EXTRACTOR_H
