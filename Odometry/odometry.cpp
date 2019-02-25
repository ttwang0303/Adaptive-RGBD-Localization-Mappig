#include "odometry.h"
#include "Core/frame.h"
#include "Utils/converter.h"
#include "generalizedicp.h"
#include "pnpsolver.h"
#include "ransac.h"

using namespace std;

Odometry::Odometry(const eAlgorithm& algorithm)
    : mOdometryAlgorithm(algorithm)
{
    if (algorithm == ADAPTIVE) {
        mpRansac = new Ransac(200, 20, 3.0f, 4);
        mpGicp = new GeneralizedICP(10, 0.07);
    } else if (algorithm == RANSAC) {
        mpRansac = new Ransac(200, 20, 3.0f, 4);
        mpGicp = nullptr;
    } else if (algorithm == ICP) {
        mpGicp = new GeneralizedICP(10, 0.07);
        mpRansac = nullptr;
    } else if (algorithm == MOTION_ONLY_BA) {
        mpBA = new PnPSolver;
        mpGicp = nullptr;
        mpRansac = nullptr;
    } else if (algorithm == ADAPTIVE_2) {
        mpBA = new PnPSolver;
        mpRansac = new Ransac(200, 20, 3.0f, 4);
        mpGicp = nullptr;
    }
}

Odometry::~Odometry()
{
    if (mpRansac)
        delete mpRansac;
    if (mpGicp)
        delete mpGicp;
    if (mpBA)
        delete mpBA;
}

void Odometry::Compute(Frame* pF1, Frame* pF2, const vector<cv::DMatch>& vMatches12)
{
    // Adaptive
    if (mOdometryAlgorithm == ADAPTIVE) {
        // Run RANSAC
        mpRansac->Iterate(pF1, pF2, vMatches12);
        cv::Mat T12;

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

        // Composition rule
        T12 = T12 * pF1->GetPose();
        pF2->SetPose(T12);

        // Update inlier flag
        for (const auto& m : mpRansac->mvInliers)
            pF2->SetInlier(m.trainIdx);

    }

    // Ransac
    else if (mOdometryAlgorithm == RANSAC) {
        mpRansac->Iterate(pF1, pF2, vMatches12);
        cv::Mat T12 = Converter::toMat<float, 4, 4>(mpRansac->mT12);

        // Composition rule
        T12 = T12 * pF1->GetPose();
        pF2->SetPose(T12);

        // Update inlier flag
        for (const auto& m : mpRansac->mvInliers)
            pF2->SetInlier(m.trainIdx);
    }

    // ICP
    else if (mOdometryAlgorithm == ICP) {
        // Not implemented yet
    }

    // Motion only Bundle Adjustment
    else if (mOdometryAlgorithm == MOTION_ONLY_BA) {
        mpBA->Compute(pF2);
    }

    // Ransac + Motion only Bundle Adjustment
    else if (mOdometryAlgorithm == ADAPTIVE_2) {
        // Get initial estimation
        mpRansac->Iterate(pF1, pF2, vMatches12);
        cv::Mat T12 = Converter::toMat<float, 4, 4>(mpRansac->mT12);

        // Composition rule
        T12 = T12 * pF1->GetPose();
        pF2->SetPose(T12);

        // Refine
        mpBA->Compute(pF2);
    }
}
