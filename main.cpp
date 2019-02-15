#include "Core/frame.h"
#include "Core/keyframe.h"
#include "Core/keyframedatabase.h"
#include "Core/landmark.h"
#include "Core/map.h"
#include "Drawer/pointclouddrawer.h"
#include "Drawer/viewer.h"
#include "Features/extractor.h"
#include "Features/matcher.h"
#include "Odometry/odometry.h"
#include "Odometry/ransac.h"
#include "Utils/common.h"
#include "Utils/converter.h"
#include "Utils/utils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <sstream>
#include <thread>

using namespace std;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_xyz/";

int main()
{
    // Good detector/descriptor combinations
    map<string, vector<string>> mCombinationsMap;
    mCombinationsMap["BRISK"] = { "BRISK", "ORB", "FREAK" };
    mCombinationsMap["FAST"] = { "SIFT" };
    mCombinationsMap["ORB"] = { "BRISK", "ORB", "FREAK" };
    mCombinationsMap["SHI_TOMASI"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SIFT", "LATCH" };
    mCombinationsMap["STAR"] = { "BRISK", "FREAK", "LATCH" };
    mCombinationsMap["SURF"] = { "BRISK", "BRIEF", "ORB", "FREAK" };

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
    Extractor* pExtractor = new Extractor(Extractor::FAST, Extractor::BRIEF, Extractor::ADAPTIVE);
    Matcher* pMatcher = new Matcher(0.9f);

    // Store Landmarks and KeyFrames
    Map* pMap = new Map();

    Database* pDatabaseKF = new Database();

    // Map and pose viewer
    PointCloudDrawer* pCloudDrawer = new PointCloudDrawer(pMap);
    Viewer* pViewer = new Viewer(pCloudDrawer);
    thread* ptViewer = new thread(&Viewer::Run, pViewer);

    // Odometry algorithm
    Odometry* pOdometry = new Odometry(Odometry::ADAPTIVE);

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Ptr<Frame> mLastFrame(new Frame);
    cv::Mat imColor, imDepth;
    for (size_t i = 0; i < nImages; i += 1) {
        imColor = cv::imread(baseDir + vImageFilenamesRGB[i], cv::IMREAD_COLOR);
        imDepth = cv::imread(baseDir + vImageFilenamesD[i], cv::IMREAD_UNCHANGED);

        Ptr<Frame> mCurrentFrame(new Frame(imColor, imDepth, vTimestamps[i]));
        mCurrentFrame->ExtractFeatures(pExtractor);

        cv::Mat Tcw;
        if (i == 0) {
            Tcw = cv::Mat::eye(4, 4, CV_32F);
        } else {
            vector<cv::DMatch> vMatches;
            pMatcher->KnnMatch(*mLastFrame, *mCurrentFrame, vMatches);

            Tcw = pOdometry->Compute(*mLastFrame, *mCurrentFrame, vMatches);

            // Composition rule
            Tcw = Tcw * mLastFrame->GetPose();
            Matcher::DrawMatches(*mLastFrame, *mCurrentFrame, pOdometry->mpRansac->mvInliers);
        }

        // Update pose
        mCurrentFrame->SetPose(Tcw);

        // Draw each 20 frames
        if (mCurrentFrame->mnId % 20 == 1 && !pOdometry->mpRansac->mvInliers.empty()) {
            KeyFrame* pKF = new KeyFrame(*mCurrentFrame, pMap, pDatabaseKF);

            // Create matched landmarks
            for (const auto& m : pOdometry->mpRansac->mvInliers) {
                int idx = m.trainIdx;
                cv::Mat x3Dw = mCurrentFrame->UnprojectWorld(idx);
                Landmark* pNewLandmark = new Landmark(x3Dw, mCurrentFrame.get(), idx);
                // mCurrentFrame->AddLandmark(pNewLandmark, idx);
                pMap->AddLandmark(pNewLandmark);
            }
            pMap->AddKeyFrame(pKF);
        }

        // Save results
        const cv::Mat& R = mCurrentFrame->GetRotationInv();
        vector<float> q = Converter::toQuaternion(R);
        const cv::Mat& t = mCurrentFrame->GetCameraCenter();
        f << setprecision(6) << mCurrentFrame->mTimestamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        cout << GREEN << ((mCurrentFrame->mnId + 1) * 100) / nImages << " %"
             << RESET << '\r';
        cout.flush();

        mLastFrame = mCurrentFrame;
    }

    f.close();
    cout << "Trajectory saved!" << endl;

    pViewer->RequestFinish();
    while (!pViewer->isFinished())
        usleep(5000);
    pangolin::BindToContext("Viewer");

    pMap->Clear();
    ptViewer->join();

    delete pExtractor;
    delete pMatcher;
    delete pMap;
    delete pDatabaseKF;
    delete pOdometry;
    delete ptViewer;
    delete pViewer;
    delete pCloudDrawer;

    return 0;
}
