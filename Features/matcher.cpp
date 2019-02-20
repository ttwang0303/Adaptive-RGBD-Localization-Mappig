#include "matcher.h"
#include "Core/frame.h"
#include "Core/keyframe.h"
#include "Core/landmark.h"
#include "extractor.h"

using namespace std;

Matcher::Matcher(float nnratio)
    : mfNNratio(nnratio)
{
    mpMatcher = cv::BFMatcher::create(Extractor::mNorm);

    if (Extractor::mNorm == cv::NORM_HAMMING) {
        TH_LOW = 50.0;
        TH_HIGH = 100.0;
    } else if (Extractor::mNorm == cv::NORM_L2) {
        // not implemented yet
    }
}

size_t Matcher::KnnMatch(Frame& pF1, Frame& pF2, vector<cv::DMatch>& vMatches12)
{
    vMatches12.clear();
    vector<vector<cv::DMatch>> matchesKnn;
    set<int> trainIdxs;

    mpMatcher->knnMatch(pF1.mDescriptors, pF2.mDescriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < mfNNratio * m2.distance) {
            if (trainIdxs.count(m1.trainIdx) > 0)
                continue;

            const size_t i1 = m1.queryIdx;
            const size_t i2 = m1.trainIdx;
            Landmark* pLM = pF1.GetLandmark(i1);
            if (!pLM)
                continue;
            if (pF1.IsOutlier(i1))
                continue;

            if (pF2.GetLandmark(i2)) {
                if (pF2.GetLandmark(i2)->Observations() > 0)
                    continue;
            }

            pF2.AddLandmark(pLM, i2);
            pF2.SetOutlier(i2);
            trainIdxs.insert(m1.trainIdx);
            vMatches12.push_back(m1);
        }
    }

    return vMatches12.size();
}

int Matcher::BoWMatch(KeyFrame* pKF1, KeyFrame* pKF2, vector<cv::DMatch>& vMatches12)
{
    DBoW3::FeatureVector::const_iterator f1it = pKF1->mFeatVec.begin();
    DBoW3::FeatureVector::const_iterator f1end = pKF1->mFeatVec.end();
    DBoW3::FeatureVector::const_iterator f2it = pKF2->mFeatVec.begin();
    DBoW3::FeatureVector::const_iterator f2end = pKF2->mFeatVec.end();

    vMatches12.reserve(pKF1->N);
    set<int> trainIdxs;

    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            // From the same word
            const vector<unsigned int> vIdx1 = f1it->second;
            const vector<unsigned int> vIdx2 = f2it->second;

            for (const auto& idx1 : vIdx1) {
                const cv::Mat& d1 = pKF1->mDescriptors.row(idx1);

                double bestDist1 = numeric_limits<double>::max();
                int bestQueryIdx1 = -1;
                int bestTrainIdx2 = -1;
                double bestDist2 = numeric_limits<double>::max();

                for (const auto& idx2 : vIdx2) {
                    const cv::Mat& d2 = pKF2->mDescriptors.row(idx2);
                    double dist = DescriptorDistance(d1, d2);

                    if (dist < bestDist1) {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestQueryIdx1 = int(idx1);
                        bestTrainIdx2 = int(idx2);
                    } else if (dist < bestDist2) {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 <= TH_LOW) {
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
                        if (trainIdxs.count(bestTrainIdx2) > 0)
                            continue;

                        cv::DMatch match12;
                        match12.queryIdx = bestQueryIdx1;
                        match12.trainIdx = bestTrainIdx2;
                        match12.distance = static_cast<float>(bestDist1);
                        vMatches12.push_back(match12);
                        trainIdxs.insert(bestTrainIdx2);
                    }
                }
            }

            f1it++;
            f2it++;
        } else if (f1it->first < f2it->first) {
            f1it = pKF1->mFeatVec.lower_bound(f2it->first);
        } else {
            f2it = pKF2->mFeatVec.lower_bound(f1it->first);
        }
    }

    return static_cast<int>(vMatches12.size());
}

void Matcher::DrawMatches(Frame& pF1, Frame& pF2, const std::vector<cv::DMatch>& m12, const int delay, const string& title)
{
    cv::Mat out;
    cv::drawMatches(pF1.mImColor, pF1.mvKeys, pF2.mImColor, pF2.mvKeys, m12, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow(title, out);
    cv::waitKey(delay);
}

void Matcher::DrawKeyPoints(Frame& pF1, const int delay)
{
    cv::Mat out;
    cv::drawKeypoints(pF1.mImColor, pF1.mvKeys, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("KeyPoints", out);
    cv::waitKey(delay);
}

void Matcher::DrawInlierPoints(Frame& frame, const int& delay, const string& title)
{
    static const float r = 5;

    cv::Mat out;
    frame.mImGray.copyTo(out);
    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < frame.N; ++i) {
        if (frame.GetLandmark(i)) {
            if (frame.IsInlier(i)) {
                const cv::KeyPoint& kp = frame.mvKeys[i];
                cv::rectangle(out, cv::Point2f(kp.pt.x - r, kp.pt.y - r), cv::Point2f(kp.pt.x + r, kp.pt.y + r),
                    cv::Scalar(0, 255, 0));
                cv::circle(out, kp.pt, 2, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    cv::imshow(title, out);
    cv::waitKey(delay);
}

double Matcher::DescriptorDistance(const cv::Mat& a, const cv::Mat& b)
{
    return cv::norm(a, b, Extractor::mNorm);
}
