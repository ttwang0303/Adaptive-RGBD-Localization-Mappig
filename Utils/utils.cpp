#include "utils.h"
#include "Core/frame.h"
#include "constants.h"
#include "converter.h"
#include <fstream>
#include <opencv2/xfeatures2d.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <sstream>

using namespace std;

void LoadImages(const string& associationFilename, vector<string>& vImageFilenamesRGB,
    vector<string>& vImageFilenamesD, vector<double>& vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(associationFilename.c_str());
    while (!fAssociation.eof()) {
        string s;
        getline(fAssociation, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vImageFilenamesD.push_back(sD);
        }
    }
}

vector<cv::DMatch> Match(Frame* pF1, Frame* pF2, cv::Ptr<cv::DescriptorMatcher> pMatcher)
{
    vector<vector<cv::DMatch>> matchesKnn;
    vector<cv::DMatch> m12;
    set<int> trainIdxs;

    pMatcher->knnMatch(pF1->mDescriptors, pF2->mDescriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < 0.9 * m2.distance) {
            if (trainIdxs.count(m1.trainIdx) > 0)
                continue;

            trainIdxs.insert(m1.trainIdx);
            m12.push_back(m1);
        }
    }

    return m12;
}

void DrawMatches(Frame* pF1, Frame* pF2, const vector<cv::DMatch>& m12, const int delay)
{
    cv::Mat out;
    cv::drawMatches(pF1->mIm, pF1->mvKps, pF2->mIm, pF2->mvKps, m12, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches", out);
    cv::waitKey(delay);
}

void DrawKeyPoints(Frame* pF1, const int delay)
{
    cv::Mat out;
    cv::drawKeypoints(pF1->mIm, pF1->mvKps, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("KeyPoints", out);
    cv::waitKey(delay);
}

cv::Ptr<cv::FeatureDetector> CreateDetector(const std::string& detector)
{
    cv::Ptr<cv::FeatureDetector> pDetector;

    if (detector == "ORB"s) {
        pDetector = cv::ORB::create(nFeatures);
    } else if (detector == "FAST"s) {
        pDetector = cv::FastFeatureDetector::create();
    } else if (detector == "HARRIS"s) {
        pDetector = cv::GFTTDetector::create(nFeatures, 0.01, 1, 3, true, 0.04);
    } else if (detector == "SHI_TOMASI"s) {
        pDetector = cv::GFTTDetector::create(nFeatures, 0.01, 10, 3, false, 0.04);
    } else if (detector == "STAR"s) {
        pDetector = cv::xfeatures2d::StarDetector::create(45, 6, 10, 10, 5);
    } else if (detector == "BRISK"s) {
        pDetector = cv::BRISK::create();
    } else if (detector == "SURF"s) {
        pDetector = cv::xfeatures2d::SURF::create();
    } else if (detector == "SIFT"s) {
        pDetector = cv::xfeatures2d::SIFT::create(nFeatures);
    }

    return pDetector;
}

cv::Ptr<cv::DescriptorExtractor> CreateDescriptor(const std::string& descriptor)
{
    cv::Ptr<cv::DescriptorExtractor> pDescriptor;

    if (descriptor == "ORB"s) {
        pDescriptor = cv::ORB::create();
    } else if (descriptor == "BRISK"s) {
        pDescriptor = cv::BRISK::create();
    } else if (descriptor == "BRIEF"s) {
        pDescriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptor == "FREAK"s) {
        pDescriptor = cv::xfeatures2d::FREAK::create();
    } else if (descriptor == "LATCH"s) {
        pDescriptor = cv::xfeatures2d::LATCH::create();
    } else if (descriptor == "SURF"s) {
        pDescriptor = cv::xfeatures2d::SURF::create();
    } else if (descriptor == "SIFT"s) {
        pDescriptor = cv::xfeatures2d::SIFT::create();
    }

    return pDescriptor;
}

bool FindHomography(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12, cv::Mat& H, vector<uchar>& vRansacStatus)
{
    vector<cv::Point2f> vSourcePoints, vTargetPoints;
    vSourcePoints.reserve(vMatches12.size());
    vTargetPoints.reserve(vMatches12.size());

    for (const auto& m : vMatches12) {
        vSourcePoints.push_back(pF1->mvKps[m.queryIdx].pt);
        vTargetPoints.push_back(pF2->mvKps[m.trainIdx].pt);
    }

    try {
        H = cv::findHomography(vSourcePoints, vTargetPoints, CV_RANSAC, 4, vRansacStatus, 500, 0.995);
        //        Mat Fundamental = findFundamentalMat(p1, p2, RansacStatus, FM_RANSAC);
        return true;
    } catch (cv::Exception& ex) {
        return false;
    }
}

cv::Mat DistanceFiler(const Frame* pF1, const Frame* pF2, vector<cv::DMatch>& vMatches12, const double thresh)
{
    cv::Mat H(3, 3, CV_64F);
    vector<uchar> vRansacStatus;
    if (!FindHomography(pF1, pF2, vMatches12, H, vRansacStatus))
        return cv::Mat();

    const auto isOutlier([&pF1, &pF2, &thresh, &H](const cv::DMatch& m12) {
        const cv::Point2f& dst = pF2->mvKps[m12.trainIdx].pt;
        const cv::Point2f& src = ApplyHomography(pF1->mvKps[m12.queryIdx].pt, H);

        double d = cv::norm(cv::Vec2f(dst.x, dst.y), cv::Vec2f(src.x, src.y));
        return d > thresh;
    });

    vMatches12.erase(remove_if(vMatches12.begin(), vMatches12.end(), isOutlier), vMatches12.end());
    return H;
}

cv::Point2f ApplyHomography(const cv::Point2f& pt, const cv::Mat& H)
{
    if (!H.empty()) {
        double d = H.at<double>(6) * pt.x + H.at<double>(7) * pt.y + H.at<double>(8);

        cv::Point2f newPt;
        newPt.x = (H.at<double>(0) * pt.x + H.at<double>(1) * pt.y + H.at<double>(2)) / d;
        newPt.y = (H.at<double>(3) * pt.x + H.at<double>(4) * pt.y + H.at<double>(5)) / d;

        return newPt;
    } else {
        return pt;
    }
}

vector<pair<double, double>> TestRecallPrecision(Frame* pF1, Frame* pF2, cv::Ptr<cv::DescriptorMatcher> pMatcher, vector<cv::DMatch>& vMatches12)
{
    vector<vector<cv::DMatch>> matchesKnn;
    vector<bool> isMatch;

    pMatcher->knnMatch(pF1->mDescriptors, pF2->mDescriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < 0.9 * m2.distance) {
            isMatch.push_back(true);
            vMatches12.push_back(m1);

        } else {
            isMatch.push_back(false);
            vMatches12.push_back(m1);
        }
    }

    vector<pair<double, double>> vDataRP;

    // Calculate Homography matrix
    vector<uchar> vRansacStatus;
    cv::Mat H(3, 3, CV_64F);
    if (!FindHomography(pF1, pF2, vMatches12, H, vRansacStatus)) {
        vDataRP.push_back({ 0, 0 });
        return vDataRP;
    }

    // Calculate maximum Euclidean distance between matches according to Homography
    vector<double> vDistances;
    double maxEuclideanDist = 0;
    for (size_t i = 0; i < vMatches12.size(); i++) {
        cv::Point2f src = pF1->mvKps[vMatches12[i].queryIdx].pt;
        cv::Point2f tgt = pF2->mvKps[vMatches12[i].trainIdx].pt;

        cv::Point2f srcT = ApplyHomography(src, H);
        double dist = cv::norm(cv::Vec2f(srcT.x, srcT.y) - cv::Vec2f(tgt.x, tgt.y));
        vDistances.push_back(dist);

        if (dist > maxEuclideanDist)
            maxEuclideanDist = dist;
    }

    // Varying the threshold between what are regarded as a correct and false positives,
    // different values of recall an precision can be calculated
    Eigen::ArrayXd threshList = Eigen::ArrayXd::LinSpaced(300, 0.0, maxEuclideanDist + 1.0);
    for (int i = 0; i < threshList.size(); ++i) {
        int positives = 0;
        int negatives = 0;
        int truePositives = 0;
        int falsePositives = 0;

        for (size_t j = 0; j < vMatches12.size(); j++) {
            if (isMatch[j]) {
                positives++;
                if (vDistances[j] < threshList(i, 0))
                    truePositives++;
            } else {
                negatives++;
                if (vDistances[j] < threshList(i, 0))
                    falsePositives++;
            }
        }

        double recall = double(truePositives) / double(positives);
        double precision = double(falsePositives) / double(negatives);
        vDataRP.push_back({ recall, precision });
    }

    return vDataRP;
}

void AddNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud, int k)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZ>);
    searchTree->setInputCloud(cloud);

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimator;
    normalEstimator.setInputCloud(cloud);
    normalEstimator.setSearchMethod(searchTree);
    normalEstimator.setKSearch(k);
    normalEstimator.compute(*normals);

    pcl::concatenateFields(*cloud, *normals, *normalsCloud);
}
