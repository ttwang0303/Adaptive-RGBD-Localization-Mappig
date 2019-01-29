#include "Core/frame.h"
#include "Utils/utils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg2_pioneer_slam/";

int main()
{
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

    map<string, vector<string>> mCombinationsMap;
    mCombinationsMap["BRISK"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["FAST"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["HARRIS"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["ORB"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["SHI_TOMASI"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["SIFT"] = { "BRISK", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["STAR"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["SURF"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };

    for (const auto& [detector, vDescriptors] : mCombinationsMap) {
        cv::Ptr<cv::FeatureDetector> pDetector = CreateDetector(detector);

        for (const auto& descriptor : vDescriptors) {
            cv::Ptr<cv::DescriptorExtractor> pDescriptor = CreateDescriptor(descriptor);
            cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::BFMatcher::create(pDescriptor->defaultNorm());

            vector<int> vStatiscs;
            cv::Mat imColor, imDepth;
            cv::TickMeter tm;

            for (size_t i = 0; i < nImages; i += 1) {
                imColor = cv::imread(baseDir + vImageFilenamesRGB[i], cv::IMREAD_COLOR);
                imDepth = cv::imread(baseDir + vImageFilenamesD[i], cv::IMREAD_UNCHANGED);

                Frame currFrame(imColor, imDepth, vTimestamps[i]);

                tm.start();
                currFrame.DetectAndCompute(pDetector, pDescriptor);
                tm.stop();

                DrawKeyPoints(&currFrame, 1);
                vStatiscs.push_back(currFrame.N);
            }

            cout << detector << "/" << descriptor << endl;
            cout << "Mean detect time: " << tm.getTimeSec() / tm.getCounter() << " s." << endl;

            auto [minIt, maxIt] = minmax_element(vStatiscs.begin(), vStatiscs.end());
            int sumKPs = accumulate(vStatiscs.begin(), vStatiscs.end(), 0);
            cout << "Max KPs: " << *maxIt << endl;
            cout << "Min KPs: " << *minIt << endl;
            cout << "Mean KPs: " << sumKPs / vStatiscs.size() << endl;
            cout << endl;
        }
    }

    return 0;
}

