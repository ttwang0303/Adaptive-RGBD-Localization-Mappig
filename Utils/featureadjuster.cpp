#include "featureadjuster.h"
#include <algorithm>
#include <cassert>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

// ------------------------------------------- DetectorAdjuster -------------------------------------------
DetectorAdjuster::DetectorAdjuster(std::string detector_name, double initial_thresh, double min_thresh,
    double max_thresh, double increase_factor, double decrease_factor)
    : thresh_(initial_thresh)
    , min_thresh_(min_thresh)
    , max_thresh_(max_thresh)
    , increase_factor_(increase_factor)
    , decrease_factor_(decrease_factor)
    , detector_name_(detector_name)
{
    if (!(detector_name_ == "SURF" || detector_name_ == "SIFT"
            || detector_name_ == "FAST" || detector_name_ == "ORB")) {
        cerr << "Unknown Descriptor: " << detector_name_ << endl;
    }
}

void DetectorAdjuster::detect(cv::InputArray image, vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    cv::Ptr<cv::Feature2D> detector;

    if (detector_name_ == "FAST")
        detector = cv::FastFeatureDetector::create(thresh_);
    else if (detector_name_ == "ORB")
        detector = cv::ORB::create(10000, 1.2, 8, 15, 0, 2, 0, 31, static_cast<int>(thresh_));
    else if (detector_name_ == "SURF")
        detector = cv::xfeatures2d::SurfFeatureDetector::create(thresh_);
    else if (detector_name_ == "SIFT")
        detector = cv::xfeatures2d::SiftFeatureDetector::create(0, 3, thresh_);
    else {
        detector = cv::FastFeatureDetector::create(thresh_);
        cerr << "Unknown detector '" << detector_name_ << "', using defaul" << endl;
    }

    detector->detect(image, keypoints, mask);
}

void DetectorAdjuster::setDecreaseFactor(double new_factor) { decrease_factor_ = new_factor; }

void DetectorAdjuster::setIncreaseFactor(double new_factor) { increase_factor_ = new_factor; }

void DetectorAdjuster::tooFew(int, int)
{
    thresh_ *= decrease_factor_;
    if (thresh_ < min_thresh_)
        thresh_ = min_thresh_;
}

void DetectorAdjuster::tooMany(int, int)
{
    thresh_ *= increase_factor_;
    if (thresh_ > max_thresh_)
        thresh_ = max_thresh_;
}

bool DetectorAdjuster::good() const { return (thresh_ > min_thresh_) && (thresh_ < max_thresh_); }

cv::Ptr<DetectorAdjuster> DetectorAdjuster::clone() const
{
    cv::Ptr<DetectorAdjuster> cloned_obj(new DetectorAdjuster(detector_name_, thresh_, min_thresh_, max_thresh_, increase_factor_, decrease_factor_));
    return cloned_obj;
}

// ----------------------------------- VideoDynamicAdaptedFeatureDetector ------------------------------------
VideoDynamicAdaptedFeatureDetector::VideoDynamicAdaptedFeatureDetector(cv::Ptr<DetectorAdjuster> a,
    int min_features, int max_features, int max_iters)
    : escape_iters_(max_iters)
    , min_features_(min_features)
    , max_features_(max_features)
    , adjuster_(a)
{
}

cv::Ptr<StatefulFeatureDetector> VideoDynamicAdaptedFeatureDetector::clone() const
{
    StatefulFeatureDetector* fd = new VideoDynamicAdaptedFeatureDetector(adjuster_->clone(),
        min_features_,
        max_features_,
        escape_iters_);
    cv::Ptr<StatefulFeatureDetector> cloned_obj(fd);
    return cloned_obj;
}

void VideoDynamicAdaptedFeatureDetector::detect(cv::InputArray _image, vector<cv::KeyPoint>& keypoints, cv::InputArray _mask)
{
    int iter_count = escape_iters_;
    bool checked_for_non_zero_mask = false;

    do {
        keypoints.clear();

        adjuster_->detect(_image, keypoints, _mask);
        int found_keypoints = static_cast<int>(keypoints.size());

        if (found_keypoints < min_features_) {
            adjuster_->tooFew(min_features_, found_keypoints);
        } else if (int(keypoints.size()) > max_features_) {
            adjuster_->tooMany(max_features_, (int)keypoints.size());
            break;
        } else
            break;

        iter_count--;
    } while (iter_count > 0 && adjuster_->good());
}

// ------------------------------------- VideoGridAdaptedFeatureDetector -------------------------------------
VideoGridAdaptedFeatureDetector::VideoGridAdaptedFeatureDetector(const cv::Ptr<StatefulFeatureDetector>& _detector,
    int _maxTotalKeypoints, int _gridRows, int _gridCols, int _edgeThreshold)
    : maxTotalKeypoints(_maxTotalKeypoints)
    , gridRows(_gridRows)
    , gridCols(_gridCols)
    , edgeThreshold(_edgeThreshold)
{
    detectors.push_back(_detector);
    while (detectors.size() < gridRows * gridCols) {
        detectors.push_back(_detector->clone());
    }
}

struct ResponseComparator {
    bool operator()(const cv::KeyPoint& a, const cv::KeyPoint& b)
    {
        return abs(a.response) > abs(b.response);
    }
};

void keepStrongest(int N, vector<cv::KeyPoint>& keypoints)
{
    if ((int)keypoints.size() > N) {
        std::vector<cv::KeyPoint>::iterator nth = keypoints.begin() + N;
        std::nth_element(keypoints.begin(), nth, keypoints.end(), ResponseComparator());
        keypoints.erase(nth, keypoints.end());
    }
}

static void aggregateKeypointsPerGridCell(std::vector<std::vector<cv::KeyPoint>>& sub_keypoint_vectors,
    std::vector<cv::KeyPoint>& keypoints_out, cv::Size gridSize, cv::Size imageSize, int edgeThreshold)
{
    for (int i = 0; i < gridSize.height; ++i) {
        int rowstart = std::max((i * imageSize.height) / gridSize.height - edgeThreshold, 0);
        for (int j = 0; j < gridSize.width; ++j) {
            int colstart = std::max((j * imageSize.width) / gridSize.width - edgeThreshold, 0);

            vector<cv::KeyPoint>& cell_keypoints = sub_keypoint_vectors[j + i * gridSize.width];
            vector<cv::KeyPoint>::iterator it = cell_keypoints.begin(), end = cell_keypoints.end();
            for (; it != end; ++it) {
                it->pt.x += colstart;
                it->pt.y += rowstart;
            }
            keypoints_out.insert(keypoints_out.end(), cell_keypoints.begin(), cell_keypoints.end());
        }
    }
}

void VideoGridAdaptedFeatureDetector::detect(cv::InputArray _image, vector<cv::KeyPoint>& keypoints, cv::InputArray _mask)
{
    cv::Mat image = _image.getMat();
    cv::Mat mask = _mask.getMat();
    std::vector<std::vector<cv::KeyPoint>> sub_keypoint_vectors(gridCols * gridRows);
    keypoints.reserve(maxTotalKeypoints);
    int maxPerCell = maxTotalKeypoints / (gridRows * gridCols);

#pragma omp parallel for
    for (int i = 0; i < gridRows; ++i) {
        int rowstart = std::max((i * image.rows) / gridRows - edgeThreshold, 0);
        int rowend = std::min(image.rows, ((i + 1) * image.rows) / gridRows + edgeThreshold);
        cv::Range row_range(rowstart, rowend);

#pragma omp parallel for
        for (int j = 0; j < gridCols; ++j) {
            int colstart = std::max((j * image.cols) / gridCols - edgeThreshold, 0);
            int colend = std::min(image.cols, ((j + 1) * image.cols) / gridCols + edgeThreshold);
            cv::Range col_range(colstart, colend);
            cv::Mat sub_image = image(row_range, col_range);
            cv::Mat sub_mask;
            if (!mask.empty()) {
                sub_mask = mask(row_range, col_range);
            }

            std::vector<cv::KeyPoint>& sub_keypoints = sub_keypoint_vectors[j + i * gridCols];
            detectors[j + i * gridCols]->detect(sub_image, sub_keypoints, sub_mask);
            keepStrongest(maxPerCell, sub_keypoints);
        }
    }

    aggregateKeypointsPerGridCell(sub_keypoint_vectors, keypoints, cv::Size(gridCols, gridRows), image.size(), edgeThreshold);
}

cv::Ptr<StatefulFeatureDetector> VideoGridAdaptedFeatureDetector::clone() const
{
    StatefulFeatureDetector* fd = new VideoGridAdaptedFeatureDetector(detectors[0]->clone(),
        maxTotalKeypoints,
        gridRows, gridCols,
        edgeThreshold);
    cv::Ptr<StatefulFeatureDetector> cloned_obj(fd);
    return cloned_obj;
}
