#include "constants.h"
#include "converter.h"
#include "frame.h"
#include "generalizedicp.h"
#include "pointclouddrawer.h"
#include "ransacpcl.h"
#include "utils.h"
#include "viewer.h"
#include <chrono>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <pangolin/pangolin.h>
#include <thread>

#define CLOUD_VIEWER 1

using namespace std;
using namespace chrono_literals;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_desk/";

int main()
{
    Viewer* pViewer = nullptr;
    thread* ptViewer = nullptr;
    PointCloudDrawer* pCloudDrawer = nullptr;

#ifdef CLOUD_VIEWER
    pCloudDrawer = new PointCloudDrawer();
    pViewer = new Viewer(pCloudDrawer);
    ptViewer = new thread(&Viewer::Run, pViewer);
#endif

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

    cv::Ptr<cv::FeatureDetector> pDetector = cv::ORB::create(1000);
    cv::Ptr<cv::DescriptorExtractor> pDescriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::DescriptorMatcher::create(pDescriptor->defaultNorm());

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Frame* prevFrame = new Frame();
    cv::Mat imColor, imDepth;

    RansacPCL ransac;

    for (size_t i = 0; i < nImages; i += 1) {
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

            ransac.Iterate(currFrame, prevFrame, vMatches12);
            vector<cv::DMatch> vInliers12;
            ransac.GetInliersDMatch(vInliers12);
            double error = ransac.ResidualError();
            cout << error << endl;

#ifdef CLOUD_VIEWER
            pCloudDrawer->AssignSourceCloud(ransac.mpTransformedCloud);
            pCloudDrawer->AssignTargetCloud(ransac.mpTargetInlierCloud);
            this_thread::sleep_for(5s);
#endif

            DrawMatches(currFrame, prevFrame, vInliers12);
            Tcw = prevFrame->mTcw * Converter::toMat<float, 4, 4>(ransac.mT12);
        }

        currFrame->SetPose(Tcw);

        // Save results
        {
            const cv::Mat& R = currFrame->mRcw;
            vector<float> q = Converter::toQuaternion(R);
            const cv::Mat& t = currFrame->mtcw;

            f << setprecision(6) << currFrame->timestamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
              << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }

        delete prevFrame;
        prevFrame = currFrame;
    }

    f.close();
    cout << "Trajectory saved!" << endl;

#ifdef CLOUD_VIEWER
    pViewer->RequestFinish();
    while (!pViewer->isFinished())
        usleep(5000);

    pangolin::BindToContext("Cloud Viewer");

    delete pCloudDrawer;
    delete pViewer;
    delete ptViewer;

#endif

    return 0;
}
