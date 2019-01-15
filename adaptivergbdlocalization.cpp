#include "adaptivergbdlocalization.h"
#include "converter.h"
#include "frame.h"
#include "generalizedicp.h"
#include "ransac.h"
#include <iostream>

using namespace std;

AdaptiveRGBDLocalization::AdaptiveRGBDLocalization()
{
    srand((long)clock());

    mCounter = 0;
    mAcumRmse = 0.0;
    T = Eigen::Matrix4f::Identity();

    ransac = new Ransac(20, 200, 3.0f, 4);
    icp = new GeneralizedICP(10, 0.05);
}

AdaptiveRGBDLocalization::~AdaptiveRGBDLocalization()
{
    delete ransac;
    delete icp;
}

cv::Mat AdaptiveRGBDLocalization::Compute(Frame* pF1, Frame* pF2, vector<cv::DMatch>& vMatches12)
{
    if (ransac->Compute(pF1, pF2, vMatches12)) {
        T = ransac->GetTransformation();
        float rmse = ransac->GetRMSE();
        mAcumRmse += rmse;
        mCounter++;

        if (rmse * 10.0f > /*std::floor*/ ((mAcumRmse / mCounter) * 10.0f)) {
            if (rmse * 10.0f > 20.0f) {
                if (icp->ComputeSubset(pF1, pF2, vMatches12))
                    T = icp->GetTransformation();
            } else {
                if (icp->Compute(pF1, pF2, vMatches12, T))
                    T = icp->GetTransformation();
            }
        }
    } else {
        cout << "Ransac fails" << endl;
        T = Eigen::Matrix4f::Identity();
    }

    return Converter::toCvMat(T);
}
