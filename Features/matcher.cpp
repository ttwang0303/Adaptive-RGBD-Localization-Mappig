#include "matcher.h"
#include "Core/frame.h"
#include "Core/keyframe.h"
#include "Core/landmark.h"
#include "Utils/common.h"
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

size_t Matcher::KnnMatch(KeyFrame* pKF1, Frame& F2, vector<cv::DMatch>& vMatches12)
{
    vector<vector<cv::DMatch>> matchesKnn;
    mpMatcher->knnMatch(pKF1->mDescriptors, F2.mDescriptors, matchesKnn, 2);

    const vector<Landmark*> vpLandmarksKF1 = pKF1->GetLandmarks();

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < mfNNratio * m2.distance) {
            size_t i1 = m1.queryIdx;
            size_t i2 = m1.trainIdx;
            Landmark* pLM = vpLandmarksKF1[i1];

            if (!pLM)
                continue;
            if (pLM->isBad())
                continue;
            if (F2.GetLandmark(i2))
                continue;

            F2.AddLandmark(pLM, i2);
            F2.SetOutlier(i2);
            vMatches12.push_back(m1);
        }
    }

    return vMatches12.size();
}

size_t Matcher::KnnMatch(Frame& pF1, Frame& pF2, vector<cv::DMatch>& vMatches12)
{
    vMatches12.clear();
    vector<vector<cv::DMatch>> matchesKnn;

    mpMatcher->knnMatch(pF1.mDescriptors, pF2.mDescriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < mfNNratio * m2.distance) {
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
            vMatches12.push_back(m1);
        }
    }

    return vMatches12.size();
}

size_t Matcher::ProjectionMatch(Frame* pFrame, const vector<Landmark*>& vpLandmarks, const float th)
{
    size_t nmatches = 0;

    for (size_t i = 0; i < vpLandmarks.size(); i++) {
        Landmark* pLM = vpLandmarks[i];
        if (!pLM->mbTrackInView)
            continue;
        if (pLM->isBad())
            continue;

        const vector<size_t> vIndices = pFrame->GetFeaturesInArea(pLM->mTrackProjX, pLM->mTrackProjY, th);
        if (vIndices.empty())
            continue;

        const cv::Mat LMd = pLM->GetDescriptor();

        double bestDist1 = numeric_limits<double>::max();
        double bestDist2 = numeric_limits<double>::max();
        int bestLevel = -1;
        int bestLevel2 = -1;
        int bestIdx = -1;

        for (auto& j : vIndices) {
            if (pFrame->GetLandmark(j)) {
                if (pFrame->GetLandmark(j)->Observations() > 0)
                    continue;
            }

            const cv::Mat& d = pFrame->mDescriptors.row(j);
            const double dist = DescriptorDistance(LMd, d);

            if (dist < bestDist1) {
                bestDist2 = bestDist1;
                bestDist1 = dist;
                bestLevel2 = bestLevel;
                bestLevel = pFrame->mvKeysUn[j].octave;
                bestIdx = j;
            } else if (dist < bestDist2) {
                bestLevel2 = pFrame->mvKeysUn[j].octave;
                bestDist2 = dist;
            }
        }

        if (bestDist1 <= TH_HIGH) {
            if (bestLevel == bestLevel2 && bestDist1 > mfNNratio * bestDist2)
                continue;

            pFrame->AddLandmark(pLM, bestIdx);
            // pFrame->SetOutlier(bestIdx);
            nmatches++;
        }
    }

    return nmatches;
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

int Matcher::Fuse(KeyFrame* pKF, const vector<Landmark*>& vpLandmarks, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();
    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused = 0;
    const size_t nLMs = vpLandmarks.size();

    for (size_t i = 0; i < nLMs; ++i) {
        Landmark* pLM = vpLandmarks[i];
        if (!pLM)
            continue;
        if (pLM->isBad() || pLM->IsInKeyFrame(pKF))
            continue;

        cv::Mat p3Dw = pLM->GetWorldPos();
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        if (p3Dc.at<float>(2) < 0.0f)
            continue;

        const float invz = 1 / p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0) * invz;
        const float y = p3Dc.at<float>(1) * invz;

        const float u = Calibration::fx * x + Calibration::cx;
        const float v = Calibration::fy * y + Calibration::cy;

        if (!pKF->IsInImage(u, v))
            continue;

        const float ur = u - Calibration::mbf * invz;

        // Search in a radius
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, th);
        if (vIndices.empty())
            continue;

        const cv::Mat dLM = pLM->GetDescriptor();

        double bestDist = numeric_limits<double>::max();
        int bestIdx = -1;

        for (auto& j : vIndices) {
            const cv::KeyPoint& kp = pKF->mvKeysUn[j];
            const int& kpLevel = kp.octave;

            if (pKF->mvuRight[j] >= 0) {
                // Check reprojection error in stereo
                const float& kpx = kp.pt.x;
                const float& kpy = kp.pt.y;
                const float& kpr = pKF->mvuRight[j];
                const float ex = u - kpx;
                const float ey = v - kpy;
                const float er = ur - kpr;
                const float e2 = ex * ex + ey * ey + er * er;

                if (e2 > 7.8f /** (kpLevel + 1)*/) {
                    continue;
                }
            } else {
                const float& kpx = kp.pt.x;
                const float& kpy = kp.pt.y;
                const float ex = u - kpx;
                const float ey = v - kpy;
                const float e2 = ex * ex + ey * ey;

                if (e2 > 5.99f /** (kpLevel + 1)*/)
                    continue;
            }

            const cv::Mat& dKF = pKF->mDescriptors.row(j);
            const double dist = DescriptorDistance(dLM, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = j;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist <= TH_LOW) {
            Landmark* pLMinKF = pKF->GetLandmark(bestIdx);
            if (pLMinKF) {
                if (!pLMinKF->isBad()) {
                    if (pLMinKF->Observations() > pLM->Observations())
                        pLM->Replace(pLMinKF);
                    else
                        pLMinKF->Replace(pLM);
                }
            } else {
                pLM->AddObservation(pKF, bestIdx);
                pKF->AddLandmark(pLM, bestIdx);
            }

            nFused++;
        }
    }

    return nFused;
}

void Matcher::DrawMatches(Frame& pF1, Frame& pF2, const std::vector<cv::DMatch>& m12, const int delay, const string& title)
{
    cv::Mat out;
    cv::drawMatches(pF1.mImGray, pF1.mvKeys, pF2.mImGray, pF2.mvKeys, m12, out, cv::Scalar::all(-1),
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
