#include "generalizedicp.h"
#include "frame.h"

using namespace std;

GeneralizedICP::GeneralizedICP()
    : GeneralizedICP(15, 0.05)
{
}

GeneralizedICP::GeneralizedICP(int iters, double maxCorrespondenceDist)
{
    mGicp.setMaximumIterations(iters);
    mGicp.setMaxCorrespondenceDistance(maxCorrespondenceDist);
    mGicp.setEuclideanFitnessEpsilon(1);
    mGicp.setTransformationEpsilon(1e-9);

    mSrcCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mTgtCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
}

bool GeneralizedICP::Compute(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12, Eigen::Matrix4f& guess)
{
    CreateCloudsFromMatches(pF1, pF2, vMatches12);
    return Align(guess);
}

bool GeneralizedICP::ComputeSubset(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12)
{
    vector<cv::DMatch> vMatchesSubset = GetSubset(vMatches12);
    CreateCloudsFromMatches(pF1, pF2, vMatchesSubset);

    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
    return Align(guess);
}

bool GeneralizedICP::Align(const Eigen::Matrix4f& guess)
{
    mGicp.setInputSource(mSrcCloud);
    mGicp.setInputTarget(mTgtCloud);

    pcl::PointCloud<pcl::PointXYZ> regCloud;
    mGicp.align(regCloud, guess);

    if (mGicp.hasConverged()) {
        mTransformation = mGicp.getFinalTransformation().matrix();
        mScore = mGicp.getFitnessScore();
        return true;
    } else {
        mTransformation = Eigen::Matrix4f::Identity();
        mScore = 1e6;
        return false;
    }
}

Eigen::Matrix4f GeneralizedICP::GetTransformation() const { return mTransformation; }

double GeneralizedICP::GetScore() const { return mScore; }

void GeneralizedICP::SetMaximumIterations(int iters) { mGicp.setMaximumIterations(iters); }

void GeneralizedICP::setMaxCorrespondenceDistance(double dist) { mGicp.setMaxCorrespondenceDistance(dist); }

void GeneralizedICP::CreateCloudsFromMatches(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12)
{
    mSrcCloud->clear();
    mTgtCloud->clear();

    for (const auto& m : vMatches12) {
        const cv::Point3f& source = pF1->kps3Dc[m.queryIdx];
        const cv::Point3f& target = pF2->kps3Dc[m.trainIdx];

        if (isnan(source.z) || isnan(target.z))
            continue;
        if (source.z <= 0 || target.z <= 0)
            continue;

        mSrcCloud->points.push_back(pcl::PointXYZ(source.x, source.y, source.z));
        mTgtCloud->points.push_back(pcl::PointXYZ(target.x, target.y, target.z));
    }

    mSrcCloud->height = 1;
    mSrcCloud->width = mSrcCloud->points.size();
    mSrcCloud->is_dense = false;

    mTgtCloud->height = 1;
    mTgtCloud->width = mTgtCloud->points.size();
    mTgtCloud->is_dense = false;
}

vector<cv::DMatch> GeneralizedICP::GetSubset(const vector<cv::DMatch>& vMatches12)
{
    set<vector<cv::DMatch>::size_type> sSampledIds;
    int safetyNet = 0;

    while (sSampledIds.size() < 0.75 * vMatches12.size()) {
        int id1 = rand() % vMatches12.size();
        int id2 = rand() % vMatches12.size();

        if (id1 > id2)
            id1 = id2;

        sSampledIds.insert(id1);

        if (++safetyNet > 10000)
            break;
    }

    vector<cv::DMatch> vSampledMatches;
    vSampledMatches.reserve(sSampledIds.size());
    for (const auto& id : sSampledIds)
        vSampledMatches.push_back(vMatches12[id]);

    return vSampledMatches;
}