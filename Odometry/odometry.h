#ifndef ODOMETRY_H
#define ODOMETRY_H

#include <opencv2/core.hpp>

class Frame;
class Ransac;
class GeneralizedICP;

class Odometry {
public:
    enum eAlgorithm {
        RANSAC = 0,
        ICP,
        ADAPTIVE
    };

    Odometry(const eAlgorithm& algorithm);
    ~Odometry();

    cv::Mat Compute(Frame* pF1, Frame* pF2, const std::vector<cv::DMatch>& vMatches12);

    Ransac* mpRansac;
    GeneralizedICP* mpGicp;

private:
    eAlgorithm mOdometryAlgorithm;
};

#endif // ODOMETRY_H
