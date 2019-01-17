#include "adaptivergbdlocalization.h"
#include "converter.h"
#include "frame.h"
#include "generalizedicp.h"
#include "ransac.h"
#include "utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_room/";

int main()
{
    vector<string> vImageFilenamesRGB;
    vector<string> vImageFilenamesD;
    vector<double> vTimestamps;
    string associationFilename = string(baseDir + "associations.txt");
    LoadImages(associationFilename, vImageFilenamesRGB, vImageFilenamesD, vTimestamps);

    int nImages = vImageFilenamesRGB.size();
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
    cv::Ptr<cv::DescriptorMatcher> pMatcher = cv::DescriptorMatcher::create("BruteForce-Hamming" /*"BruteForce"*/);

    ofstream f("CameraTrajectory.txt");
    f << fixed;
    Frame* prevFrame = new Frame();
    cv::Mat imColor, imDepth;

    AdaptiveRGBDLocalization odometry(AdaptiveRGBDLocalization::RANSAC);

    for (int i = 0; i < nImages; i += 1) {
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
            Tcw = odometry.Compute(currFrame, prevFrame, vMatches12);
            DrawMatches(currFrame, prevFrame, odometry.ransac->GetMatches());
            Tcw = prevFrame->mTcw * Tcw;

            if (!odometry.hasConverged())
                cout << "Fail" << endl;
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

    return 0;
}

// pcl::visualization::PCLVisualizer viewer("v");
/*PointCloud::Ptr out(new PointCloud);
pcl::transformPointCloud(*prevFrame->cloud, *out, T);

viewer.removePointCloud("tgt");
viewer.removePointCloud("trans");
viewer.addPointCloud(currFrame->cloud, "tgt");
viewer.addPointCloud(out, "trans");
viewer.spinOnce();*/
