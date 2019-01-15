#ifndef ADAPTIVERGBDLOCALIZATION_H
#define ADAPTIVERGBDLOCALIZATION_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>

class Ransac;
class GeneralizedICP;
class Frame;

class AdaptiveRGBDLocalization {
public:
    AdaptiveRGBDLocalization();

    ~AdaptiveRGBDLocalization();

    cv::Mat Compute(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& vMatches12);

    Ransac* ransac;
    GeneralizedICP* icp;

private:
    float mAcumRmse;
    int mCounter;
    Eigen::Matrix4f T;
};

#endif // ADAPTIVERGBDLOCALIZATION_H
