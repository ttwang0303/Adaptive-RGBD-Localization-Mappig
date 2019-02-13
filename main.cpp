#include "Core/frame.h"
#include "Core/landmark.h"
#include "Core/map.h"
#include "Drawer/pointclouddrawer.h"
#include "Drawer/viewer.h"
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

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/living_room_traj1_frei_png/";

int main()
{
    // Good detector/descriptor combinations
    map<string, vector<string>> mCombinationsMap;
    mCombinationsMap["BRISK"] = { "BRISK", "ORB", "FREAK" };
    mCombinationsMap["FAST"] = { "SIFT" };
    mCombinationsMap["ORB"] = { "BRISK", "ORB", "FREAK" };
    mCombinationsMap["SHI_TOMASI"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SIFT", "LATCH" };
    mCombinationsMap["STAR"] = { "BRISK", "FREAK", "LATCH" };
    mCombinationsMap["SURF"] = { "BRISK", "ORB", "FREAK" };

    // Read files
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

    // This is a Feature-based method
    cv::Ptr<cv::FeatureDetector> pDetector(CreateAdaptiveDetector("SURF"));
    cv::Ptr<cv::DescriptorExtractor> pDescriptor(CreateDescriptor("BRIEF"));
    cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::BFMatcher::create(pDescriptor->defaultNorm());

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Frame* prevFrame = nullptr;
    cv::Mat imColor, imDepth;

    // Store Landmarks and KeyFrames
    Map* pMap = new Map();

    // Map a pose viewer
    PointCloudDrawer* pCloudDrawer = new PointCloudDrawer(pMap);
    Viewer* pViewer = new Viewer(pCloudDrawer);
    thread* ptViewer = new thread(&Viewer::Run, pViewer);

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

            // Refine with ICP
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

            // Composition rule
            Tcw = Tcw * prevFrame->GetPose();
            DrawMatches(prevFrame, currFrame, sac.mvInliers);
        }

        // Update pose
        currFrame->SetPose(Tcw);

        // Draw each 20 frames
        if (currFrame->mnId % 20 == 1 && !sac.mvInliers.empty()) {
            currFrame->mbIsKeyFrame = true;

            // Create matched landmarks
            for (const auto& m : sac.mvInliers) {
                int idx = m.trainIdx;
                cv::Mat x3Dw = currFrame->UnprojectWorld(idx);
                Landmark* pNewLandmark = new Landmark(x3Dw, currFrame, idx);
                pNewLandmark->AddObservation(currFrame, idx);
                currFrame->AddLandmark(pNewLandmark, idx);
                pMap->AddLandmark(pNewLandmark);
            }
            pMap->AddKeyFrame(currFrame);
        }

        // Save results
        const cv::Mat& R = currFrame->GetRotationInv();
        vector<float> q = Converter::toQuaternion(R);
        const cv::Mat& t = currFrame->GetCameraCenter();
        f << setprecision(6) << currFrame->mTimestamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        if (prevFrame) {
            if (!prevFrame->mbIsKeyFrame)
                delete prevFrame;
        }
        prevFrame = currFrame;
    }

    f.close();
    cout << "Trajectory saved!" << endl;

    pViewer->RequestFinish();
    while (!pViewer->isFinished())
        usleep(5000);
    pangolin::BindToContext("Cloud Viewer");

    pMap->Clear();

    return 0;
}
