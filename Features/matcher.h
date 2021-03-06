#ifndef MATCHER_H
#define MATCHER_H

#include <opencv2/features2d.hpp>

class Frame;
class KeyFrame;
class Landmark;

class Matcher {
public:
    Matcher(float nnratio = 0.6f);

    static double DescriptorDistance(const cv::Mat& a, const cv::Mat& b);

    size_t KnnMatch(KeyFrame* pKF1, Frame& F2, std::vector<cv::DMatch>& vMatches12);

    size_t KnnMatch(Frame& pF1, Frame& pF2, std::vector<cv::DMatch>& vMatches12);

    size_t ProjectionMatch(Frame* pFrame, const std::vector<Landmark*>& vpLandmarks, const float th = 3.0);

    int BoWMatch(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<cv::DMatch>& vMatches12);

    // Project Landmarks into pKF and search for duplicated
    int Fuse(KeyFrame* pKF, const std::vector<Landmark*>& vpLandmarks, const float th = 3.0);

    static void DrawMatches(Frame& pF1, Frame& pF2, const std::vector<cv::DMatch>& m12, const int delay = 1, const std::string& title = "Matches");
    static void DrawKeyPoints(Frame& pF1, const int delay = 1);
    static void DrawInlierPoints(Frame& frame, const int& delay = 1, const std::string& title = "Inlier Matches");

private:
    float mfNNratio;

    cv::Ptr<cv::DescriptorMatcher> mpMatcher;

    double TH_LOW;
    double TH_HIGH;
};

#endif // MATCHER_H
