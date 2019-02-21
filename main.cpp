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
#include "Odometry/pnpsolver.h"
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
const string vocDir = "./vocab/voc_fr1_ORB_ORB.yml.gz";

int main()
{
    srand((long)clock());

    // Good detector/descriptor combinations
    map<string, vector<string>> mCombinationsMap;
    mCombinationsMap["BRISK"] = { "BRISK", "ORB", "FREAK" };
    mCombinationsMap["FAST"] = { "SIFT", "BRIEF" };
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
    Extractor* pExtractor = new Extractor(Extractor::ORB, Extractor::ORB, Extractor::ADAPTIVE);
    Matcher* pMatcher = new Matcher(0.9f);

    // Store Landmarks and KeyFrames
    Map* pMap = new Map();

    // Place recognition (Loop detection)
    cout << "Loading vocabulary...";
    cout.flush();
    DBoW3::Vocabulary* pVocabulary = new DBoW3::Vocabulary(vocDir);
    if (pVocabulary->empty()) {
        cout << "Wrong vocab path!" << endl;
        terminate();
    }
    cout << " done." << endl;
    Database* pDatabaseKF = new Database(pVocabulary);

    // Map and pose viewer
    PointCloudDrawer* pCloudDrawer = new PointCloudDrawer(pMap);
    Viewer* pViewer = new Viewer(pCloudDrawer);
    thread* ptViewer = new thread(&Viewer::Run, pViewer);

    // Odometry algorithm
    Odometry* pOdometry = new Odometry(Odometry::RANSAC);

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Frame mLastFrame;
    cv::Mat imColor, imDepth;
    cv::TickMeter tm;
    KeyFrame* pKFref;

    for (size_t n = 0; n < nImages; n += 1) {
        imColor = cv::imread(baseDir + vImageFilenamesRGB[n], cv::IMREAD_COLOR);
        imDepth = cv::imread(baseDir + vImageFilenamesD[n], cv::IMREAD_UNCHANGED);

        tm.start();
        Frame mCurrentFrame(imColor, imDepth, vTimestamps[n]);
        mCurrentFrame.ExtractFeatures(pExtractor);

        // Initialization
        if (n == 0) {
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

            KeyFrame* pKFini = new KeyFrame(mCurrentFrame, pMap, pDatabaseKF);
            pMap->AddKeyFrame(pKFini);
            pKFref = pKFini;

            // Create Lamdmarks
            for (size_t i = 0; i < mCurrentFrame.N; ++i) {
                if (mCurrentFrame.mvKeys3Dc[i].z > 0) {
                    cv::Mat x3Dw = mCurrentFrame.UnprojectWorld(i);
                    Landmark* pNewLM = new Landmark(x3Dw, pKFini, pMap);
                    pNewLM->AddObservation(pKFini, i);
                    pKFini->AddLandmark(pNewLM, i);
                    pNewLM->ComputeDistinctiveDescriptors();
                    pMap->AddLandmark(pNewLM);

                    mCurrentFrame.AddLandmark(pNewLM, i);
                }
            }
            cout << "Map created with " << pMap->LandmarksInMap() << " points" << endl;
        }
        // Track
        else {
            if (n > 1)
                UpdateLastFrame(mLastFrame, pMap);

            // Feature matching
            vector<cv::DMatch> vMatches;
            size_t nmatches = pMatcher->KnnMatch(mLastFrame, mCurrentFrame, vMatches);
            pOdometry->Compute(mLastFrame, mCurrentFrame, vMatches);

            Matcher::DrawInlierPoints(mCurrentFrame);

            // Discard outliers
            for (size_t i = 0; i < mCurrentFrame.N; ++i) {
                Landmark* pLM = mCurrentFrame.GetLandmark(i);
                if (!pLM)
                    continue;
                if (mCurrentFrame.IsInlier(i))
                    continue;

                mCurrentFrame.AddLandmark(static_cast<Landmark*>(nullptr), i);
                // mCurrentFrame.SetInlier(i);
                pLM->mbTrackInView = false;
                pLM->mnLastFrameSeen = mCurrentFrame.GetId();
            }
        }

        // Check if its necessary to insert a new KF
        if (NeedNewKF(pKFref, mCurrentFrame)) {
            KeyFrame* pKF = new KeyFrame(mCurrentFrame, pMap, pDatabaseKF);
            pKFref = pKF;

            pDatabaseKF->Add(pKF);
            cout << pMap->KeyFramesInMap() << " KFs in map" << endl;
        }

        // Save results
        const cv::Mat& R = mCurrentFrame.GetRotationInv();
        vector<float> q = Converter::toQuaternion(R);
        const cv::Mat& t = mCurrentFrame.GetCameraCenter();
        f << setprecision(6) << mCurrentFrame.mTimestamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        cout << GREEN << ((mCurrentFrame.GetId() + 1) * 100) / nImages << " %"
             << RESET << '\r';
        cout.flush();

        mLastFrame = mCurrentFrame;
        tm.stop();
    }

    f.close();
    cout << "Trajectory saved!" << endl;
    cout << "Mean tracking time: " << tm.getTimeSec() / tm.getCounter() << " s." << endl;

    pViewer->RequestFinish();
    while (!pViewer->isFinished())
        usleep(5000);
    pangolin::BindToContext("Viewer");

    pDatabaseKF->Clear();
    pMap->Clear();

    ptViewer->join();

    delete pExtractor;
    delete pMatcher;
    delete pDatabaseKF;
    delete pOdometry;
    delete ptViewer;
    delete pViewer;
    delete pCloudDrawer;
    delete pMap;

    return 0;
}
