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

        const cv::KeyPoint& kpU = frame.mvKeysUn[i];
        cv::Mat Xw = pLM->GetWorldPos();

        v2D.push_back(kpU.pt);
        v3D.push_back(cv::Point3f(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2)));
        vnIndex.push_back(i);
    }

    if (v2D.size() < 10)
        return 0;

    cv::Mat r, t, inliers, T;
    bool bOK = cv::solvePnPRansac(v3D, v2D, frame.mK, cv::Mat(), r, t, false, 500, 3.0f, 0.85, inliers);

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
