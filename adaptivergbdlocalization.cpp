#include "adaptivergbdlocalization.h"
#include "converter.h"
#include "frame.h"
#include "generalizedicp.h"
#include "ransac.h"
#include <iostream>

using namespace std;

AdaptiveRGBDLocalization::AdaptiveRGBDLocalization()
    : mAlgorithm(RANSAC_ICP)
    , mStatus(false)
    , T(Eigen::Matrix4f::Identity())
    , mMiu1(5.0)
    , mMiu2(20.0)
{
    srand((long)clock());

    ransac = new Ransac(20, 200, 3.0f, 4);
    icp = new GeneralizedICP(15, 0.05);
}

AdaptiveRGBDLocalization::AdaptiveRGBDLocalization(const AdaptiveRGBDLocalization::Algorithm algorithm)
    : mAlgorithm(algorithm)
    , mStatus(false)
    , T(Eigen::Matrix4f::Identity())
    , mMiu1(5.0)
    , mMiu2(20.0)
{
    srand((long)clock());

    switch (mAlgorithm) {
    case RANSAC:
        ransac = new Ransac(20, 200, 3.0f, 4);
        icp = nullptr;
        break;

    case ICP:
        icp = new GeneralizedICP(15, 0.05);
        ransac = nullptr;
        break;

    case RANSAC_ICP:
        ransac = new Ransac(20, 200, 3.0f, 4);
        icp = new GeneralizedICP(15, 0.05);
        break;
    }
}

AdaptiveRGBDLocalization::~AdaptiveRGBDLocalization()
{
    if (ransac)
        delete ransac;

    if (icp)
        delete icp;
}

cv::Mat AdaptiveRGBDLocalization::Compute(Frame* pF1, Frame* pF2, vector<cv::DMatch>& vMatches12)
{
    switch (mAlgorithm) {
    case RANSAC:
        ComputeRansac(pF1, pF2, vMatches12);
        break;

    case ICP:
        ComputeICP(pF1, pF2, vMatches12);
        break;

    case RANSAC_ICP:
        ComputeAdaptive(pF1, pF2, vMatches12);
        break;
    }

    return Converter::toCvMat(T);
}

void AdaptiveRGBDLocalization::SetGuess(const Eigen::Matrix4f& guess) { T = guess; }

void AdaptiveRGBDLocalization::SetMiu1(const float miu1)
{
    if (miu1 < mMiu2)
        mMiu1 = miu1;
}

void AdaptiveRGBDLocalization::SetMiu2(const float miu2)
{
    if (miu2 > mMiu1)
        mMiu2 = miu2;
}

bool AdaptiveRGBDLocalization::hasConverged() const { return mStatus == true; }

void AdaptiveRGBDLocalization::ComputeRansac(Frame* pF1, Frame* pF2, vector<cv::DMatch>& vMatches12)
{
    if (ransac->Compute(pF1, pF2, vMatches12)) {
        T = ransac->GetTransformation();
        mStatus = true;
    } else {
        T = Eigen::Matrix4f::Identity();
        mStatus = false;
    }
}

void AdaptiveRGBDLocalization::ComputeICP(Frame* pF1, Frame* pF2, vector<cv::DMatch>& vMatches12)
{
    if (icp->Compute(pF1, pF2, vMatches12, T)) {
        T = icp->GetTransformation();
        mStatus = true;
    } else {
        T = Eigen::Matrix4f::Identity();
        mStatus = false;
    }
}

void AdaptiveRGBDLocalization::ComputeAdaptive(Frame* pF1, Frame* pF2, vector<cv::DMatch>& vMatches12)
{
    if (ransac->Compute(pF1, pF2, vMatches12)) {
        T = ransac->GetTransformation();
        float rmse = ransac->GetRMSE();
        mStatus = true;

        if (rmse * 10.0f > mMiu2) {
            if (icp->ComputeSubset(pF1, pF2, vMatches12))
                T = icp->GetTransformation();
        } else if (rmse * 10.0f > mMiu1) {
            if (icp->Compute(pF1, pF2, vMatches12, T))
                T = icp->GetTransformation();
        }

    } else {
        T = Eigen::Matrix4f::Identity();
        mStatus = false;
    }
}
