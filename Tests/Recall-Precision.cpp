#include "Core/frame.h"
#include "Core/landmark.h"
#include "Utils/utils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sstream>

using namespace std;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/Feature_tests/wall_viewpoint/";

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
    for (int i = 0; i < 6; ++i) {
        stringstream ss;
        ss << "img" << i + 1 << ".ppm";
        vImageFilenamesRGB.push_back(ss.str());
    }
    size_t nImages = vImageFilenamesRGB.size();

    cout << "Start processing sequence: " << baseDir
         << "\nImages in the sequence: " << nImages << endl
         << endl;

    for (const auto& [detector, vDescriptors] : mCombinationsMap) {
        cv::Ptr<cv::FeatureDetector> pDetector = CreateDetector(detector);

        for (const auto& descriptor : vDescriptors) {
            cout << detector << "-" << descriptor << endl;
            cv::Ptr<cv::DescriptorExtractor> pDescriptor = CreateDescriptor(descriptor);
            cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::BFMatcher::create(pDescriptor->defaultNorm());

            cv::Mat imColor;
            cv::TickMeter tm;
            stringstream ss;
            ss << "RP_" << detector << "-" << descriptor << ".csv";
            ofstream f(ss.str());

            // Prepare base Frame
            imColor = cv::imread(baseDir + vImageFilenamesRGB[0], cv::IMREAD_COLOR);
            Frame* pBaseFrame = new Frame(imColor);
            pBaseFrame->DetectAndCompute(pDetector, pDescriptor);
            cout << "Base KPs: " << pBaseFrame->N << endl;

            // Prepare reference Frame
            imColor = cv::imread(baseDir + vImageFilenamesRGB[4], cv::IMREAD_COLOR);
            Frame* pReferenceFrame = new Frame(imColor);
            pReferenceFrame->DetectAndCompute(pDetector, pDescriptor);
            cout << "Reference KPs: " << pReferenceFrame->N << endl
                 << endl;

            // Recall/Precision test
            vector<cv::DMatch> vMatchesBR;
            vector<pair<double, double>> vDataRP = TestRecallPrecision(pBaseFrame, pReferenceFrame, pMatcher, vMatchesBR);
            DrawMatches(pBaseFrame, pReferenceFrame, vMatchesBR, 1);

            for (vector<pair<double, double>>::iterator it = vDataRP.begin(); it != vDataRP.end(); it++) {
                // recall - precision
                f << it->second << ", " << it->first << endl;
            }

            f.close();
        }
    }

    return 0;
}
