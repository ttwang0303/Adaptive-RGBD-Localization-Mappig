#ifndef MATCHER_H
#define MATCHER_H

#include <opencv2/features2d.hpp>

class Frame;

class Matcher {
public:
    Matcher(float nnratio = 0.6f);

    void KnnMatch(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& vMatches12);

    static void DrawMatches(Frame* pF1, Frame* pF2, const std::vector<cv::DMatch>& m12, const int delay = 1);
    static void DrawKeyPoints(Frame* pF1, const int delay = 1);

private:
    float mfNNratio;

    cv::Ptr<cv::DescriptorMatcher> mpMatcher;
};

#endif // MATCHER_H
