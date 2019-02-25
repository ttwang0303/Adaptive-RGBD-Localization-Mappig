#include "Core/keyframedatabase.h"
#include "Core/map.h"
#include "Drawer/mapdrawer.h"
#include "Drawer/viewer.h"
#include "Features/extractor.h"
#include "Odometry/odometry.h"
#include "System/localmapping.h"
#include "System/tracking.h"
#include "Utils/utils.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <pangolin/pangolin.h>
#include <thread>

using namespace std;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_desk/";
const string vocDir = "./vocab/voc_fr1_GFTT_BRIEF.yml.gz";

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
    Extractor* pExtractor = new Extractor(Extractor::GFTT, Extractor::BRIEF, Extractor::NORMAL);

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
    cout << *pVocabulary << endl;
    Database* pKeyFrameDB = new Database(pVocabulary);

    // Map and pose viewer
    MapDrawer* pMapDrawer = new MapDrawer(pMap);
    Viewer* pViewer = new Viewer(pMapDrawer);
    thread* ptViewer = new thread(&Viewer::Run, pViewer);

    // Odometry algorithm
    Odometry* pOdometry = new Odometry(Odometry::ADAPTIVE_2);

    LocalMapping* pLocalMapper = new LocalMapping(pMap, pVocabulary);
    thread* ptLocalMapping = new thread(&LocalMapping::Run, pLocalMapper);

    Tracking* pTracker = new Tracking(pVocabulary, pMap, pKeyFrameDB, pExtractor);
    pTracker->SetLocalMapper(pLocalMapper);
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

    pLocalMapper->RequestFinish();
    if (pViewer) {
        pViewer->RequestFinish();
        while (!pViewer->isFinished())
            usleep(5000);
    }

    while (!pLocalMapper->isFinished()) {
        usleep(5000);
    }

    pangolin::BindToContext("Viewer");

    pKeyFrameDB->Clear();

    ptViewer->join();
    ptLocalMapping->join();

    pTracker->SaveTrajectory("CameraTrajectory.txt");
    pTracker->SaveKeyFrameTrajectory("KeyFrameTrajectory.txt");

    pMap->Clear();

    delete pTracker;
    delete pExtractor;
    delete pOdometry;
    delete ptViewer;
    delete pViewer;
    delete pMapDrawer;
    delete ptLocalMapping;
    delete pLocalMapper;
    delete pKeyFrameDB;
    delete pVocabulary;
    delete pMap;

    return 0;
}
