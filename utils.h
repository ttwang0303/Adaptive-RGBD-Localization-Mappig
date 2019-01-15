#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Frame;

void LoadImages(const std::string& associationFilename, std::vector<std::string>& vImageFilenamesRGB,
    std::vector<std::string>& vImageFilenamesD, std::vector<double>& vTimestamps);

std::vector<cv::DMatch> Match(Frame* pF1, Frame* pF2, cv::Ptr<cv::DescriptorMatcher> pMatcher);
void DrawMatches(Frame* pF1, Frame* pF2, const std::vector<cv::DMatch>& m12, const int delay = 1);

#endif // UTILS_H
