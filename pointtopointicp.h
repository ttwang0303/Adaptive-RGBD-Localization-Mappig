#ifndef POINTTOPOINTICP_H
#define POINTTOPOINTICP_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>

class Frame;

class PointToPointICP {
public:
    PointToPointICP();

    Eigen::Matrix4f Compute(const Frame* srcFrame, const Frame* dstFrame, const std::vector<cv::DMatch>& vMatches, Eigen::Matrix4f& guess);

    void CreateCorrespondeces(const Frame* srcFrame, const Frame* dstFrame, std::vector<cv::DMatch>& vMatches);
};

#endif // POINTTOPOINTICP_H
