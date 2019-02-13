#include "matcher.h"
#include "Core/frame.h"
#include "extractor.h"

using namespace std;

Matcher::Matcher(float nnratio)
    : mfNNratio(nnratio)
{
    mpMatcher = cv::BFMatcher::create(Extractor::mNorm);
}

void Matcher::KnnMatch(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& vMatches12)
{
    vMatches12.clear();
    vector<vector<cv::DMatch>> matchesKnn;
    set<int> trainIdxs;

    mpMatcher->knnMatch(pF1->mDescriptors, pF2->mDescriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < mfNNratio * m2.distance) {
            if (trainIdxs.count(m1.trainIdx) > 0)
                continue;

            trainIdxs.insert(m1.trainIdx);
            vMatches12.push_back(m1);
        }
    }
}

void Matcher::DrawMatches(Frame* pF1, Frame* pF2, const std::vector<cv::DMatch>& m12, const int delay)
{
    cv::Mat out;
    cv::drawMatches(pF1->mIm, pF1->mvKps, pF2->mIm, pF2->mvKps, m12, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches", out);
    cv::waitKey(delay);
}

void Matcher::DrawKeyPoints(Frame* pF1, const int delay)
{
    cv::Mat out;
    cv::drawKeypoints(pF1->mIm, pF1->mvKps, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("KeyPoints", out);
    cv::waitKey(delay);
}
