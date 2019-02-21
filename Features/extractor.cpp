#include "extractor.h"
#include "Utils/common.h"
#include "detectoradjuster.h"
#include "statefulfeaturedetector.h"
#include "videodynamicadaptedfeaturedetector.h"
#include "videogridadaptedfeaturedetector.h"
#include <iostream>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

int Extractor::mNorm;

Extractor::Extractor(const Extractor::eAlgorithm& detector, const Extractor::eAlgorithm& descriptor, const Extractor::eMode& mode)
{
    mDetectorAlgorithm = detector;
    mDescriptorAlgorithm = descriptor;
    mMode = mode;

    if (mode == ADAPTIVE) {
        if (detector == FAST || detector == SURF || detector == SIFT || detector == ORB) {
            CreateAdaptiveDetector();
            cout << " - Adaptive Feature Extraction" << endl;
        } else {
            cerr << "Not supported adaptive detector" << endl;
            terminate();
        }
    } else if (mode == NORMAL) {
        CreateDetector();
        cout << " - Normal Feature Extraction" << endl;
    }

    CreateDescriptor();

    mNorm = mpDescriptor->defaultNorm();
}

void Extractor::Extract(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    mpDetector->detect(image, keypoints);
    if (keypoints.size() > nFeatures)
        cv::KeyPointsFilter::retainBest(keypoints, nFeatures);

    mpDescriptor->compute(image, keypoints, descriptors);
}

void Extractor::CreateAdaptiveDetector()
{
    DetectorAdjuster* pAdjuster = nullptr;

    if (mDetectorAlgorithm == FAST)
        pAdjuster = new DetectorAdjuster(mDetectorAlgorithm, 20, 2, 10000, 1.3, 0.7);
    else if (mDetectorAlgorithm == SURF)
        pAdjuster = new DetectorAdjuster(mDetectorAlgorithm, 200, 2, 10000, 1.3, 0.7);
    else if (mDetectorAlgorithm == SIFT)
        pAdjuster = new DetectorAdjuster(mDetectorAlgorithm, 0.04, 0.0001, 10000, 1.3, 0.7);
    else if (mDetectorAlgorithm == ORB)
        pAdjuster = new DetectorAdjuster(mDetectorAlgorithm, 20, 2, 10000, 1.3, 0.7);

    const int minFeatures = 600;
    const int maxFeatures = minFeatures * 1.7;
    const int iterations = 5;
    const int edgeTh = 31;

    const int gridResolution = 3;
    int gridCells = gridResolution * gridResolution;
    int gridMin = round(minFeatures / static_cast<float>(gridCells));
    int gridMax = round(maxFeatures / static_cast<float>(gridCells));

    StatefulFeatureDetector* pDetector = new VideoDynamicAdaptedFeatureDetector(pAdjuster, gridMin, gridMax, iterations);
    mpDetector.reset(new VideoGridAdaptedFeatureDetector(pDetector, maxFeatures, gridResolution, gridResolution, edgeTh));
}

void Extractor::CreateDetector()
{
    switch (mDetectorAlgorithm) {
    case ORB:
        mpDetector = cv::ORB::create(nFeatures, 1.2f, 8);
        break;
    case FAST:
        mpDetector = cv::FastFeatureDetector::create();
        break;
    case GFTT:
        mpDetector = cv::GFTTDetector::create(nFeatures, 0.01, 5, 3, false, 0.04);
        break;
    case STAR:
        mpDetector = cv::xfeatures2d::StarDetector::create(45, 6, 10, 10, 5);
        break;
    case BRISK:
        mpDetector = cv::BRISK::create(30, 8, 1.0f);
        break;
    case SURF:
        mpDetector = cv::xfeatures2d::SURF::create(100, 4, 3, false, false);
        break;
    case SIFT:
        mpDetector = cv::xfeatures2d::SIFT::create(nFeatures);
        break;
    default:
        cerr << "Invalid detector" << endl;
        terminate();
    }
}

void Extractor::CreateDescriptor()
{
    if (mDescriptorAlgorithm == ORB)
        mpDescriptor = cv::ORB::create();
    else if (mDescriptorAlgorithm == BRISK)
        mpDescriptor = cv::BRISK::create();
    else if (mDescriptorAlgorithm == BRIEF)
        mpDescriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    else if (mDescriptorAlgorithm == FREAK)
        mpDescriptor = cv::xfeatures2d::FREAK::create();
    else if (mDescriptorAlgorithm == SURF)
        mpDescriptor = cv::xfeatures2d::SURF::create();
    else if (mDescriptorAlgorithm == SIFT)
        mpDescriptor = cv::xfeatures2d::SIFT::create();
    else if (mDescriptorAlgorithm == LATCH)
        mpDescriptor = cv::xfeatures2d::LATCH::create();
    else {
        cerr << "Invlaid descriptor" << endl;
        terminate();
    }
}
