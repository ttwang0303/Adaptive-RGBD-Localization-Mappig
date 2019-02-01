#include "Core/frame.h"
#include "Odometry/generalizedicp.h"
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

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_desk/";

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

    cv::Ptr<cv::FeatureDetector> pDetector = CreateDetector("FAST");
    cv::Ptr<cv::DescriptorExtractor> pDescriptor = CreateDescriptor("SIFT");
    cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::BFMatcher::create(pDescriptor->defaultNorm());

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Frame* prevFrame = nullptr;
    cv::Mat imColor, imDepth;
    vector<float> vResidualStatistics;

    GeneralizedICP icp(15, 0.05);

    for (size_t i = 0; i < nImages; i += 1) {
        imColor = cv::imread(baseDir + vImageFilenamesRGB[i], cv::IMREAD_COLOR);
        imDepth = cv::imread(baseDir + vImageFilenamesD[i], cv::IMREAD_UNCHANGED);

        Frame* currFrame = new Frame(imColor, imDepth, vTimestamps[i]);
        currFrame->DetectAndCompute(pDetector, pDescriptor);
        currFrame->CreateCloud();
        currFrame->VoxelGridFilterCloud(0.03f);

        cv::Mat Tcw;
        if (i == 0) {
            Tcw = cv::Mat::eye(4, 4, CV_32F);
        } else {
            // Run ICP
            bool bOK = icp.Compute(prevFrame, currFrame, icp.mT12);

            if (bOK) {
                vResidualStatistics.push_back(icp.mScore);

                // Get transformation estimation
                Tcw = Converter::toMat<float, 4, 4>(icp.mT12);

                // Update pose
                Tcw = Tcw * prevFrame->mTcw;
            } else {
                cout << "ICP fail" << endl;
                Tcw = cv::Mat::eye(4, 4, CV_32F);
                Tcw = Tcw * prevFrame->mTcw;
            }
        }

        currFrame->SetPose(Tcw);
        cout << "Frame: " << currFrame->mnId << endl;

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

    auto [minIt, maxIt] = std::minmax_element(vResidualStatistics.begin(), vResidualStatistics.end());
    float sum = std::accumulate(vResidualStatistics.begin(), vResidualStatistics.end(), 0.0f);
    cout << "Max residual: " << *maxIt << endl;
    cout << "Min residual: " << *minIt << endl;
    cout << "Mean residual: " << sum / static_cast<float>(vResidualStatistics.size()) << endl;

    f.close();
    cout << "Trajectory saved!" << endl;

    return 0;
}

