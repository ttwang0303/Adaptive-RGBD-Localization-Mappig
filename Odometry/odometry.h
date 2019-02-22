#ifndef ODOMETRY_H
#define ODOMETRY_H

#include <opencv2/core.hpp>

class Frame;
class Ransac;
class GeneralizedICP;
class PnPSolver;

class Odometry {
public:
    enum eAlgorithm {
        RANSAC = 0,
        ICP,
        MOTION_ONLY_BA,
        ADAPTIVE,
        ADAPTIVE_2
    };

    Odometry(const eAlgorithm& algorithm);
    ~Odometry();

    void Compute(Frame *pF1, Frame *pF2, const std::vector<cv::DMatch>& vMatches12);

    Ransac* mpRansac;
    GeneralizedICP* mpGicp;
    PnPSolver* mpBA;

private:
    eAlgorithm mOdometryAlgorithm;
};

#endif // ODOMETRY_H
