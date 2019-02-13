#include "Core/frame.h"
#include "Odometry/generalizedicp.h"
#include "Odometry/ransac.h"
#include "Utils/constants.h"
#include "Utils/converter.h"
#include "Utils/featureadjuster.h"
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

    //    cv::Ptr<cv::FeatureDetector> pDetector = CreateDetector("FAST");
    //    cv::Ptr<cv::DescriptorExtractor> pDescriptor = CreateDescriptor("ORB");
    //    cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::BFMatcher::create(pDescriptor->defaultNorm());

    cv::Ptr<cv::FeatureDetector> pDetector(CreateDetector2("FAST"));
    cv::Ptr<cv::DescriptorExtractor> pDescriptor(CreateDescriptor2("ORB"));
    cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::BFMatcher::create(pDescriptor->defaultNorm());

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Frame* prevFrame = nullptr;
    cv::Mat imColor, imDepth;
    cv::Mat lastT12;

    Ransac sac(200, 20, 3.0f, 4);
    GeneralizedICP icp(20, 0.07);

    for (size_t i = 0; i < nImages; i += 1) {
        imColor = cv::imread(baseDir + vImageFilenamesRGB[i], cv::IMREAD_COLOR);
        imDepth = cv::imread(baseDir + vImageFilenamesD[i], cv::IMREAD_UNCHANGED);

        Frame* currFrame = new Frame(imColor, imDepth, vTimestamps[i]);
        currFrame->DetectAndCompute(pDetector, pDescriptor);

        cv::Mat Tcw;
        if (i == 0) {
            Tcw = cv::Mat::eye(4, 4, CV_32F);
        } else {
            vector<cv::DMatch> vMatches = Match(prevFrame, currFrame, pMatcher);

            // Run RANSAC
            sac.Iterate(prevFrame, currFrame, vMatches);

            if (sac.mvInliers.size() < 20 || sac.rmse * 10.0f >= 7.0f) {
                if (sac.rmse * 10.0f >= 20) {

                    if (icp.Compute(sac.mpSourceCloud, sac.mpTargetCloud, Eigen::Matrix4f::Identity()))
                        Tcw = Converter::toMat<float, 4, 4>(icp.mT12);
                    else
                        Tcw = cv::Mat::eye(4, 4, CV_32F);

                } else {
                    if (icp.Compute(sac.mpSourceCloud, sac.mpTargetCloud, sac.mT12))
                        Tcw = Converter::toMat<float, 4, 4>(icp.mT12);
                    else
                        Tcw = Converter::toMat<float, 4, 4>(sac.mT12);
                }
            } else {
                Tcw = Converter::toMat<float, 4, 4>(sac.mT12);
            }

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
