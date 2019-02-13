#include "detectoradjuster.h"
#include <iostream>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

DetectorAdjuster::DetectorAdjuster(const Extractor::eAlgorithm& detector, double initialThresh, double minThresh,
    double maxThresh, double increaseFactor, double decreaseFactor)
    : mThresh(initialThresh)
    , mMinThresh(minThresh)
    , mMaxThresh(maxThresh)
    , mIncreaseFactor(increaseFactor)
    , mDecreaseFactor(decreaseFactor)
    , mDetectorAlgorithm(detector)
{
    if (!(detector == Extractor::eAlgorithm::SURF || detector == Extractor::eAlgorithm::SIFT
            || detector == Extractor::eAlgorithm::FAST || detector == Extractor::eAlgorithm::ORB)) {
        cerr << "Not supported detector" << endl;
    }
}

void DetectorAdjuster::detect(cv::InputArray image, vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    cv::Ptr<cv::Feature2D> detector;

    if (mDetectorAlgorithm == Extractor::eAlgorithm::FAST)
        detector = cv::FastFeatureDetector::create(mThresh);
    else if (mDetectorAlgorithm == Extractor::eAlgorithm::ORB)
        detector = cv::ORB::create(10000, 1.2f, 8, 15, 0, 2, 0, 31, static_cast<int>(mThresh));
    else if (mDetectorAlgorithm == Extractor::eAlgorithm::SURF)
        detector = cv::xfeatures2d::SurfFeatureDetector::create(mThresh);
    else if (mDetectorAlgorithm == Extractor::eAlgorithm::SIFT)
        detector = cv::xfeatures2d::SiftFeatureDetector::create(0, 3, mThresh);

    detector->detect(image, keypoints, mask);
}

void DetectorAdjuster::setDecreaseFactor(double new_factor) { mDecreaseFactor = new_factor; }

void DetectorAdjuster::setIncreaseFactor(double new_factor) { mIncreaseFactor = new_factor; }

void DetectorAdjuster::tooFew(int, int)
{
    mThresh *= mDecreaseFactor;
    if (mThresh < mMinThresh)
        mThresh = mMinThresh;
}

void DetectorAdjuster::tooMany(int, int)
{
    mThresh *= mIncreaseFactor;
    if (mThresh > mMaxThresh)
        mThresh = mMaxThresh;
}

bool DetectorAdjuster::good() const
{
    return (mThresh > mMinThresh) && (mThresh < mMaxThresh);
}

cv::Ptr<DetectorAdjuster> DetectorAdjuster::clone() const
{
    cv::Ptr<DetectorAdjuster> pNewObject(new DetectorAdjuster(mDetectorAlgorithm, mThresh, mMinThresh, mMaxThresh, mIncreaseFactor, mDecreaseFactor));
    return pNewObject;
}
