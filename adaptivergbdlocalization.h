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
    enum Algorithm {
        RANSAC = 1,
        ICP,
        RANSAC_ICP
    };

    AdaptiveRGBDLocalization();

    AdaptiveRGBDLocalization(const Algorithm algorithm);

    ~AdaptiveRGBDLocalization();

    cv::Mat Compute(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& vMatches12);

    void SetGuess(const Eigen::Matrix4f& guess);

    void SetMiu1(const float miu1);
    void SetMiu2(const float miu2);

    bool hasConverged() const;

    Ransac* ransac;
    GeneralizedICP* icp;

private:
    void ComputeRansac(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& vMatches12);
    void ComputeICP(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& vMatches12);
    void ComputeAdaptive(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& vMatches12);

    Algorithm mAlgorithm;
    bool mStatus;
    Eigen::Matrix4f T;
    float mMiu1, mMiu2;
};

#endif // ADAPTIVERGBDLOCALIZATION_H
