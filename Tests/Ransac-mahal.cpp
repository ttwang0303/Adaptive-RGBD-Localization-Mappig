#include "Core/frame.h"
#include "Drawer/pointclouddrawer.h"
#include "Drawer/viewer.h"
#include "Odometry/generalizedicp.h"
#include "Odometry/ransac.h"
#include "Utils/constants.h"
#include "Utils/converter.h"
#include "Utils/utils.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <pangolin/pangolin.h>
#include <thread>

using namespace std;
using namespace chrono_literals;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_room/";

int main()
{
    PointCloudDrawer* pCloudDrawer = new PointCloudDrawer();
    Viewer* pViewer = new Viewer(pCloudDrawer);
    thread* ptViewer = new thread(&Viewer::Run, pViewer);

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

    cv::Ptr<cv::FeatureDetector> pDetector = cv::xfeatures2d::SURF::create();
    cv::Ptr<cv::DescriptorExtractor> pDescriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::DescriptorMatcher::create(pDescriptor->defaultNorm());

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Frame* prevFrame = nullptr;
    cv::Mat imColor, imDepth;
    vector<float> vResidualStatistics;

    Ransac ransac(500, 20, 3.0f, 4);
    ransac.CheckDepth(false);

    for (size_t i = 0; i < nImages; i += 3) {
        imColor = cv::imread(baseDir + vImageFilenamesRGB[i], cv::IMREAD_COLOR);
        imDepth = cv::imread(baseDir + vImageFilenamesD[i], cv::IMREAD_UNCHANGED);

        Frame* currFrame = new Frame(imColor, imDepth, vTimestamps[i]);
        currFrame->DetectAndCompute(pDetector, pDescriptor);

        cv::Mat Tcw;
        if (i == 0) {
            Tcw = cv::Mat::eye(4, 4, CV_32F);
        } else {
            // curr -> prev
            vector<cv::DMatch> vMatches12 = Match(currFrame, prevFrame, pMatcher);

            // Run RANSAC
            ransac.Iterate(currFrame, prevFrame, vMatches12);

            // Draw Inliers matches
            DrawMatches(currFrame, prevFrame, ransac.mvInliers);

            // Get transformation estimation
            Tcw = Converter::toMat<float, 4, 4>(ransac.mT12);

            // Update pose
            Tcw = prevFrame->mTcw * Tcw;

            // Display clouds
            vResidualStatistics.push_back(ransac.TransformSourcePointCloud());
            pCloudDrawer->UpdateSourceCloud(ransac.mpTransformedCloud);
            pCloudDrawer->UpdateTargetCloud(ransac.mpTargetInlierCloud);
            this_thread::sleep_for(1s);
        }

        currFrame->SetPose(Tcw);

        // Save results
        const cv::Mat& R = currFrame->mRcw;
        vector<float> q = Converter::toQuaternion(R);
        const cv::Mat& t = currFrame->mtcw;
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

    pViewer->RequestFinish();
    while (!pViewer->isFinished())
        usleep(5000);

    pangolin::BindToContext("Cloud Viewer");

    return 0;
}
