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
#include "System/tracking.h"
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
    Database* pKeyFrameDB = new Database(pVocabulary);

    // Map and pose viewer
    PointCloudDrawer* pCloudDrawer = new PointCloudDrawer(pMap);
    Viewer* pViewer = new Viewer(pCloudDrawer);
    thread* ptViewer = new thread(&Viewer::Run, pViewer);

    // Odometry algorithm
    Odometry* pOdometry = new Odometry(Odometry::RANSAC);

    Tracking* pTracker = new Tracking(pVocabulary, pMap, pKeyFrameDB, pExtractor);
    pTracker->SetOdometer(pOdometry);

    cv::Mat imColor, imDepth;
    cv::TickMeter tm;

    for (size_t n = 0; n < nImages; n += 1) {
        imColor = cv::imread(baseDir + vImageFilenamesRGB[n], cv::IMREAD_COLOR);
        imDepth = cv::imread(baseDir + vImageFilenamesD[n], cv::IMREAD_UNCHANGED);

        tm.start();
        pTracker->Track(imColor, imDepth, vTimestamps[n]);
        tm.stop();
    }

    cout << "Mean tracking time: " << tm.getTimeSec() / tm.getCounter() << " s." << endl;

    pViewer->RequestFinish();
    while (!pViewer->isFinished())
        usleep(5000);
    pangolin::BindToContext("Viewer");

    pKeyFrameDB->Clear();
    pMap->Clear();

    ptViewer->join();

    delete pExtractor;
    delete pMatcher;
    delete pKeyFrameDB;
    delete pOdometry;
    delete ptViewer;
    delete pViewer;
    delete pCloudDrawer;
    delete pMap;

    return 0;
}
