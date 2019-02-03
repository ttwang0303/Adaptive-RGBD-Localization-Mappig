#include "Core/frame.h"
#include "Odometry/generalizedicp.h"
#include "Odometry/kabsch.h"
#include "Odometry/ransac.h"
#include "Utils/constants.h"
#include "Utils/converter.h"
#include "Utils/utils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <pangolin/pangolin.h>
#include <sstream>
#include <thread>

using namespace std;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_room/";

int main()
{
    map<string, vector<string>> mCombinationsMap;
    mCombinationsMap["BRISK"] = { "BRISK", "ORB", "FREAK" };
    mCombinationsMap["FAST"] = { "SIFT" };
    mCombinationsMap["ORB"] = { "BRISK", "ORB", "FREAK" };
    mCombinationsMap["SHI_TOMASI"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SIFT", "LATCH" };
    mCombinationsMap["STAR"] = { "BRISK", "FREAK", "LATCH" };
    mCombinationsMap["SURF"] = { "BRISK", "ORB", "FREAK" };

    vector<string> vImageFilenamesRGB;
    vector<string> vImageFilenamesD;
    vector<double> vTimestamps;
    string associationFilename = string(baseDir + "associations.txt");
    LoadImages(associationFilename, vImageFilenamesRGB, vImageFilenamesD, vTimestamps);

    size_t nImages = vImageFilenamesRGB.size();
    if (vImageFilenamesRGB.empty()) {
        cerr << "\nNo images found in provided path." << endl;
        return 1;
    } else if (vImageFilenamesD.size() != vImageFilenamesRGB.size()) {
        cerr << "\nDifferent number of images for rgb and depth." << endl;
        return 1;
    }

    cout << "Start processing sequence: " << baseDir
         << "\nImages in the sequence: " << nImages << endl
         << endl;

    cv::Ptr<cv::FeatureDetector> pDetector = CreateDetector("SURF");
    cv::Ptr<cv::DescriptorExtractor> pDescriptor = CreateDescriptor("ORB");
    cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::BFMatcher::create(pDescriptor->defaultNorm());

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Frame* prevFrame = nullptr;
    cv::Mat imColor, imDepth;
    cv::Mat lastT12;

    Ransac sac(200, 20, 3.0f, 4);
    GeneralizedICP icp(20, 0.05);

    for (size_t i = 0; i < nImages; i += 1) {
        imColor = cv::imread(baseDir + vImageFilenamesRGB[i], cv::IMREAD_COLOR);
        imDepth = cv::imread(baseDir + vImageFilenamesD[i], cv::IMREAD_UNCHANGED);

        Frame* currFrame = new Frame(imColor, imDepth, vTimestamps[i]);
        currFrame->GridDetectAndCompute(pDetector, pDescriptor, 1, 1);

        cv::Mat Tcw;
        if (i == 0) {
            Tcw = cv::Mat::eye(4, 4, CV_32F);
        } else {
            vector<cv::DMatch> vMatches = Match(prevFrame, currFrame, pMatcher);

            // Run RANSAC
            sac.Iterate(prevFrame, currFrame, vMatches);

            // Get inliers
            Eigen::MatrixXf setSrc(sac.mvInliers.size(), 3);
            Eigen::MatrixXf setTgt(sac.mvInliers.size(), 3);
            for (int i = 0; i < sac.mvInliers.size(); i++) {
                const auto& m = sac.mvInliers[i];
                const cv::Point3f& source = prevFrame->mvKps3Dc[m.queryIdx];
                const cv::Point3f& target = currFrame->mvKps3Dc[m.trainIdx];

                setSrc(i, 0) = source.x;
                setSrc(i, 1) = source.y;
                setSrc(i, 2) = source.z;

                setTgt(i, 0) = target.x;
                setTgt(i, 1) = target.y;
                setTgt(i, 2) = target.z;
            }

            Kabsch kabsch;
            Eigen::Matrix4f trans = kabsch.Compute(setSrc, setTgt);

            cout << sac.mT12 << endl;
            cout << trans << endl;
            cout << endl;

            Tcw = Converter::toMat<float, 4, 4>(sac.mT12);
            lastT12 = Tcw;
            Tcw = Tcw * prevFrame->mTcw;
            DrawMatches(prevFrame, currFrame, sac.mvInliers);
        }

        currFrame->SetPose(Tcw);

        // Save results
        const cv::Mat& R = currFrame->mRwc;
        vector<float> q = Converter::toQuaternion(R);
        const cv::Mat& t = currFrame->mOw;
        f << setprecision(6) << currFrame->mTimestamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        if (prevFrame)
            delete prevFrame;

        prevFrame = currFrame;
    }

    f.close();
    cout << "Trajectory saved!" << endl;

    return 0;
}
