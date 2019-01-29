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

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/Feature_tests/wall/";

int main()
{
    map<string, vector<string>> mCombinationsMap;
    mCombinationsMap["BRISK"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "LATCH" };
    mCombinationsMap["FAST"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["ORB"] = { "BRISK", "ORB", "BRIEF", "FREAK", "LATCH" };
    mCombinationsMap["SHI_TOMASI"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "SIFT", "LATCH" };
    mCombinationsMap["STAR"] = { "BRISK", "ORB", "BRIEF", "FREAK", "SURF", "LATCH" };
    mCombinationsMap["SURF"] = { "BRISK", "ORB", "BRIEF", "FREAK" };

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
            cv::Ptr<cv::DescriptorExtractor> pDescriptor = CreateDescriptor(descriptor);
            cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::BFMatcher::create(pDescriptor->defaultNorm());

            cv::Mat imColor;
            cv::TickMeter tm;
            stringstream ss;
            ss << "Landmarks_" << detector << "-" << descriptor << ".csv";
            ofstream f(ss.str());

            // Prepare base Frame and create Landmarks and associate to Frame
            imColor = cv::imread(baseDir + vImageFilenamesRGB[0], cv::IMREAD_COLOR);
            Frame* pBaseFrame = new Frame(imColor);
            pBaseFrame->DetectAndCompute(pDetector, pDescriptor);

            for (int i = 0; i < pBaseFrame->N; ++i) {
                Landmark* pNewLandmark = new Landmark(pBaseFrame, i);
                pNewLandmark->AddObservation(pBaseFrame, i);
                pBaseFrame->AddLandmark(pNewLandmark, i);
            }

            cout << detector << "-" << descriptor << endl;
            cout << "Initial Landmarks: " << pBaseFrame->N << endl;

            for (size_t i = 1; i < nImages; i += 1) {
                imColor = cv::imread(baseDir + vImageFilenamesRGB[i], cv::IMREAD_COLOR);
                Frame* pReferenceFrame = new Frame(imColor);
                pReferenceFrame->DetectAndCompute(pDetector, pDescriptor);

                vector<cv::DMatch> vMatchesBR = Match(pBaseFrame, pReferenceFrame, pMatcher);
                DistanceFiler(pBaseFrame, pReferenceFrame, vMatchesBR, 2.0);

                // Asign landmarks seen between base and reference frame
                {
                    vector<cv::DMatch> vGoodMatch;
                    for (const auto& m : vMatchesBR) {
                        Landmark* pLandmark = pBaseFrame->mvpLandmarks[m.queryIdx];

                        if (pReferenceFrame->mvpLandmarks[m.trainIdx]) {
                            if (pReferenceFrame->mvpLandmarks[m.trainIdx]->Observations() > 0)
                                continue;
                        }

                        pLandmark->AddObservation(pReferenceFrame, m.trainIdx);
                        pReferenceFrame->AddLandmark(pLandmark, m.trainIdx);
                        vGoodMatch.push_back(m);
                    }
                    vMatchesBR = vGoodMatch;
                }

                // cout << pBaseFrame->mnId + 1 << "-" << pReferenceFrame->mnId + 1 << " -> " << vMatchesBR.size() << endl;
            }

            std::map<size_t, size_t> histogram;
            for (const auto pLandmark : pBaseFrame->mvpLandmarks) {
                if (pLandmark->Observations() > 1)
                    histogram[pLandmark->Observations()]++;
            }

            for (const auto& [obs, numLandmarks] : histogram) {
                cout << obs << " - " << numLandmarks << endl;
                f << obs << "," << numLandmarks << endl;
            }
            cout << endl;

            f.close();
        }
    }

    return 0;
}

