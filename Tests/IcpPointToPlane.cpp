#include "Core/frame.h"
#include "Odometry/generalizedicp.h"
#include "Odometry/icp/icpPointToPlane.h"
#include "Odometry/icp/icpPointToPoint.h"
#include "Odometry/icp/matrix.h"
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
        currFrame->DetectAndCompute(pDetector, pDescriptor);

        cv::Mat Tcw;
        if (i == 0) {
            Tcw = cv::Mat::eye(4, 4, CV_32F);
        } else {
            vector<cv::DMatch> vMatches = Match(prevFrame, currFrame, pMatcher);

            // Run RANSAC
            sac.Iterate(prevFrame, currFrame, vMatches);

            int32_t dim = 3;
            int32_t num = sac.mvInliers.size();
            double* Model = (double*)calloc(3 * num, sizeof(double));
            double* Template = (double*)calloc(3 * num, sizeof(double));

            for (size_t i = 0; i < num; ++i) {
                const cv::DMatch& m = sac.mvInliers[i];

                // soruce
                Template[i * 3 + 0] = static_cast<double>(prevFrame->mvKps3Dc[m.queryIdx].x);
                Template[i * 3 + 1] = static_cast<double>(prevFrame->mvKps3Dc[m.queryIdx].y);
                Template[i * 3 + 2] = static_cast<double>(prevFrame->mvKps3Dc[m.queryIdx].z);

                // target
                Model[i * 3 + 0] = static_cast<double>(currFrame->mvKps3Dc[m.trainIdx].x);
                Model[i * 3 + 1] = static_cast<double>(currFrame->mvKps3Dc[m.trainIdx].y);
                Model[i * 3 + 2] = static_cast<double>(currFrame->mvKps3Dc[m.trainIdx].z);
            }

            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = sac.mT12.block<3, 3>(0, 0);
            Eigen::Vector3f trans = sac.mT12.block<3, 1>(0, 3);
            float* ptr = rot.data();
            Matrix R = Matrix(3, 3, ptr);
            ptr = trans.data();
            Matrix t = Matrix(3, 1, ptr);

            IcpPointToPlane icp(Model, num, dim, 15, 5.0);
            // IcpPointToPoint icp(Model, num, dim);
            icp.setMaxIterations(50);
            icp.setMinDeltaParam(1e-8);
            double res = icp.fit(Template, num, R, t, 0.01);

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    sac.mT12(i, j) = R.val[i][j];
            }

            for (int i = 0; i < 3; ++i)
                sac.mT12(i, 3) = t.val[i][0];

            // free memory
            free(Model);
            free(Template);

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

