#include "ransacpcl.h"
#include "Core/frame.h"
#include "Utils/converter.h"
#include <boost/make_shared.hpp>

using namespace std;

RansacPCL::RansacPCL()
{
    mpSampleConsensus = boost::make_shared<pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>>();
    mpSampleConsensus->setInlierThreshold(0.05);
    mpSampleConsensus->setMaximumIterations(500);
    mpSampleConsensus->setRefineModel(true);

    mpSourceCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpTargetCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpTransformedCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpSourceInlierCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpTargetInlierCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    mT12 = Eigen::Matrix4f::Identity();
}

void RansacPCL::Iterate(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12)
{
    CreateCloudFromMatches(pF1, pF2, vMatches12);

    mpSampleConsensus->setInputSource(mpSourceCloud);
    mpSampleConsensus->setInputTarget(mpTargetCloud);
    mpSampleConsensus->getRemainingCorrespondences(mCorrespondences, mInliersCorrespondences);

    mT12 = mpSampleConsensus->getBestTransformation();
}

void RansacPCL::CreateCloudFromMatches(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12)
{
    mCorrespondences.clear();
    mvDMatches.clear();
    mInliersCorrespondences.clear();
    mpSourceCloud->points.clear();
    mpTargetCloud->points.clear();
    size_t idx = 0;

    mCorrespondences.reserve(vMatches12.size());
    mvDMatches.reserve(vMatches12.size());
    mpSourceCloud->points.reserve(vMatches12.size());
    mpTargetCloud->points.reserve(vMatches12.size());

    for (const auto& m : vMatches12) {
        const cv::Point3f& source = pF1->mvKeys3Dc[m.queryIdx];
        const cv::Point3f& target = pF2->mvKeys3Dc[m.trainIdx];

        if (isnan(source.z) || isnan(target.z))
            continue;
        if (source.z <= 0 || target.z <= 0)
            continue;

        mpSourceCloud->points.push_back(pcl::PointXYZ(source.x, source.y, source.z));
        mpTargetCloud->points.push_back(pcl::PointXYZ(target.x, target.y, target.z));

        pcl::Correspondence match(idx, idx, m.distance);
        mCorrespondences.push_back(match);
        mvDMatches.push_back(m);
        idx++;
    }
}

void RansacPCL::GetInliersDMatch(std::vector<cv::DMatch>& vInliers)
{
    vInliers.clear();
    vInliers.resize(mInliersCorrespondences.size());

    for (size_t i = 0; i < mInliersCorrespondences.size(); ++i) {
        int idx = mInliersCorrespondences[i].index_query;
        vInliers[i] = mvDMatches[idx];
    }
}

float RansacPCL::ResidualError()
{
    mpTransformedCloud->points.clear();
    mpSourceInlierCloud->points.clear();
    mpTargetInlierCloud->points.clear();

    size_t N = mInliersCorrespondences.size();
    float d = 0.0;
    Eigen::Matrix3f R12 = mT12.block(0, 0, 3, 3);
    Eigen::Vector3f t12(mT12(0, 3), mT12(1, 3), mT12(2, 3));

    mpTransformedCloud->points.reserve(N);
    mpSourceInlierCloud->points.reserve(N);
    mpTargetInlierCloud->points.reserve(N);

    for (const auto& inlier : mInliersCorrespondences) {
        int idxSrc = inlier.index_query;
        int idxTgt = inlier.index_match;

        mpSourceInlierCloud->points.push_back(mpSourceCloud->points[idxSrc]);
        mpTargetInlierCloud->points.push_back(mpTargetCloud->points[idxTgt]);

        Eigen::Vector3f src(mpSourceCloud->points[idxSrc].x, mpSourceCloud->points[idxSrc].y, mpSourceCloud->points[idxSrc].z);
        Eigen::Vector3f tgt(mpTargetCloud->points[idxTgt].x, mpTargetCloud->points[idxTgt].y, mpTargetCloud->points[idxTgt].z);

        Eigen::Vector3f trans = R12 * src + t12;
        mpTransformedCloud->points.push_back(pcl::PointXYZ(trans.x(), trans.y(), trans.z()));

        d += (trans - tgt).norm();
    }

    return d / N;
}

void RansacPCL::SetInlierThreshold(double thresh)
{
    mpSampleConsensus->setInlierThreshold(thresh);
}

void RansacPCL::SetMaximumIterations(int iters)
{
    mpSampleConsensus->setMaximumIterations(iters);
}
