#include "pnpransac.h"
#include "Core/frame.h"
#include "Core/landmark.h"
#include "Utils/common.h"
#include "Utils/converter.h"
#include <opencv2/calib3d.hpp>
#include <vector>

using namespace std;

int PnPRansac::Compute(Frame& frame)
{
    vector<cv::Point2f> v2D;
    vector<cv::Point3f> v3D;
    vector<size_t> vnIndex;

    for (size_t i = 0; i < frame.N; ++i) {
        Landmark* pLM = frame.GetLandmark(i);
        if (!pLM)
            continue;

        const cv::KeyPoint& kp = frame.mvKeys[i];
        cv::Mat Xw = pLM->GetWorldPos();

        v2D.push_back(kp.pt);
        v3D.push_back(cv::Point3f(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2)));
        vnIndex.push_back(i);
    }

    if (v2D.size() < 10)
        return 0;

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = Calibration::fx;
    K.at<float>(1, 1) = Calibration::fy;
    K.at<float>(0, 2) = Calibration::cx;
    K.at<float>(1, 2) = Calibration::cy;
    cv::Mat r, t, inliers, T;
    bool bOK = cv::solvePnPRansac(v3D, v2D, K, cv::Mat(), r, t, false, 500, 2.0f, 0.85, inliers);

    if (bOK) {
        T = Converter::toHomogeneous(r, t);
        frame.SetPose(T);

        for (int i = 0; i < inliers.rows; ++i) {
            int n = inliers.at<int>(i);
            const size_t idx = vnIndex[n];
            frame.SetInlier(idx);
        }
    } else {
        cerr << "PnPRansac fail" << endl;
        terminate();
    }

    return inliers.rows;
}
