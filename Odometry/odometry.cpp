#include "odometry.h"
#include "Utils/converter.h"
#include "generalizedicp.h"
#include "ransac.h"

using namespace std;

Odometry::Odometry(const eAlgorithm& algorithm)
    : mOdometryAlgorithm(algorithm)
{
    if (algorithm == ADAPTIVE) {
        mpRansac = new Ransac(200, 20, 3.0f, 4);
        mpGicp = new GeneralizedICP(20, 0.07);
    } else if (algorithm == RANSAC) {
        mpRansac = new Ransac(200, 20, 3.0f, 4);
        mpGicp = nullptr;
    } else if (algorithm == ICP) {
        mpGicp = new GeneralizedICP(20, 0.07);
        mpRansac = nullptr;
    }
}

Odometry::~Odometry()
{
    if (mpRansac)
        delete mpRansac;
    if (mpGicp)
        delete mpGicp;
}

cv::Mat Odometry::Compute(Frame& pF1, Frame& pF2, const vector<cv::DMatch>& vMatches12)
{
    cv::Mat T12;

    if (mOdometryAlgorithm == ADAPTIVE) {
        // Run RANSAC
        mpRansac->Iterate(&pF1, &pF2, vMatches12);

        // Refine with ICP
        if (mpRansac->mvInliers.size() < 20 || mpRansac->rmse * 10.0f >= 7.0f) {
            if (mpRansac->rmse * 10.0f >= 20) {

                if (mpGicp->Compute(mpRansac->mpSourceCloud, mpRansac->mpTargetCloud, Eigen::Matrix4f::Identity()))
                    T12 = Converter::toMat<float, 4, 4>(mpGicp->mT12);
                else
                    T12 = cv::Mat::eye(4, 4, CV_32F);

            } else {
                if (mpGicp->Compute(mpRansac->mpSourceCloud, mpRansac->mpTargetCloud, mpRansac->mT12))
                    T12 = Converter::toMat<float, 4, 4>(mpGicp->mT12);
                else
                    T12 = Converter::toMat<float, 4, 4>(mpRansac->mT12);
            }
        } else {
            T12 = Converter::toMat<float, 4, 4>(mpRansac->mT12);
        }
    } else if (mOdometryAlgorithm == RANSAC) {
        mpRansac->Iterate(&pF1, &pF2, vMatches12);
        T12 = Converter::toMat<float, 4, 4>(mpRansac->mT12);
    } else if (mOdometryAlgorithm == ICP) {
        // Not implemented yet
    }

    return T12;
}
