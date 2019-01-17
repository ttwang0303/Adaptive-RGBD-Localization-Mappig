#include "ransac.h"
#include "frame.h"
#include <pcl/common/transformation_from_correspondences.h>

using namespace std;

Ransac::Ransac()
    : Ransac(20, 200, 3.0f, 4)
{
}

Ransac::Ransac(uint minInlierTh, int iters, float maxMahalanobisDist, uint sampleSize)
    : mMinInlierTh(minInlierTh)
    , mIterations(iters)
    , mMaxMahalanobisDistance(maxMahalanobisDist)
    , mSampleSize(sampleSize)
{
}

bool Ransac::Compute(Frame* pF1, Frame* pF2, vector<cv::DMatch>& m12)
{
    if (m12.size() < mMinInlierTh)
        return false;

    uint minInlierTh;
    if (mMinInlierTh > 0.75 * m12.size())
        minInlierTh = 0.75 * m12.size();
    else
        minInlierTh = mMinInlierTh;

    mvMatches.clear();
    rmse = 1e6;
    mTransformation = Eigen::Matrix4f::Identity();
    bool validTf = false;
    int realIters = 0;
    int validIters = 0;
    double inlierError;

    sort(m12.begin(), m12.end());

    for (int n = 0; (n < mIterations && m12.size() >= mSampleSize); n++) {
        double refinedError = 1e6;
        vector<cv::DMatch> vRefinedMatches;
        vector<cv::DMatch> vInlierMatches = SampleMatches(m12);
        Eigen::Matrix4f refinedTransformation = Eigen::Matrix4f::Identity();
        realIters++;

        for (int refinements = 1; refinements < 20; refinements++) {
            Eigen::Matrix4f transformation = GetTransformFromMatches(pF1, pF2, vInlierMatches, validTf);

            if (!validTf && transformation != transformation)
                break;

            inlierError = ComputeInliersAndError(pF1, pF2, m12, transformation, vInlierMatches);

            if (vInlierMatches.size() < minInlierTh || inlierError > mMaxMahalanobisDistance)
                break;

            if (vInlierMatches.size() >= vRefinedMatches.size() && inlierError <= refinedError) {
                size_t prevNumInliers = vRefinedMatches.size();
                assert(inlierError >= 0);
                refinedTransformation = transformation;
                vRefinedMatches = vInlierMatches;
                refinedError = inlierError;

                if (vInlierMatches.size() == prevNumInliers)
                    break;
            } else {
                break;
            }
        }

        if (vRefinedMatches.size() > 0) {
            validIters++;

            if (refinedError <= rmse && vRefinedMatches.size() >= mvMatches.size() && vRefinedMatches.size() >= minInlierTh) {
                rmse = refinedError;
                mTransformation = refinedTransformation;
                mvMatches.assign(vRefinedMatches.begin(), vRefinedMatches.end());

                if (vRefinedMatches.size() > m12.size() * 0.5)
                    n += 10;
                if (vRefinedMatches.size() > m12.size() * 0.75)
                    n += 10;
                if (vRefinedMatches.size() > m12.size() * 0.8)
                    break;
            }
        }
    }

    if (validIters == 0) {
        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
        vector<cv::DMatch> vInlierMatches;
        inlierError = ComputeInliersAndError(pF1, pF2, m12, transformation, vInlierMatches);

        if (vInlierMatches.size() > minInlierTh && inlierError < mMaxMahalanobisDistance) {
            assert(inlierError >= 0);
            mTransformation = transformation;
            mvMatches.assign(vInlierMatches.begin(), vInlierMatches.end());
            rmse += inlierError;
            validIters++;
        }
    }

    bool enoughAbsolute = mvMatches.size() >= minInlierTh;
    return enoughAbsolute;
}

vector<cv::DMatch> Ransac::SampleMatches(const vector<cv::DMatch>& vMatches)
{
    set<vector<cv::DMatch>::size_type> sSampledIds;
    int safetyNet = 0;

    while (sSampledIds.size() < mSampleSize && vMatches.size() >= mSampleSize) {
        int id1 = rand() % vMatches.size();
        int id2 = rand() % vMatches.size();

        if (id1 > id2)
            id1 = id2;

        sSampledIds.insert(id1);

        if (++safetyNet > 10000)
            break;
    }

    vector<cv::DMatch> vSampledMatches;
    vSampledMatches.reserve(sSampledIds.size());
    for (const auto& id : sSampledIds)
        vSampledMatches.push_back(vMatches[id]);

    return vSampledMatches;
}

Eigen::Matrix4f Ransac::GetTransformFromMatches(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches, bool& valid)
{
    pcl::TransformationFromCorrespondences tfc;
    valid = true;
    float weight = 1.0;

    for (const auto& m : vMatches) {
        const cv::Point3f& from = pF1->kps3Dc[m.queryIdx];
        const cv::Point3f& to = pF2->kps3Dc[m.trainIdx];

        if (isnan(from.z) || isnan(to.z))
            continue;

        weight = 1.0 / (from.z * to.z);
        tfc.add(Eigen::Vector3f(from.x, from.y, from.z), Eigen::Vector3f(to.x, to.y, to.z), weight);
    }

    return tfc.getTransformation().matrix();
}

double Ransac::ComputeInliersAndError(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& m12, const Eigen::Matrix4f& transformation4f, std::vector<cv::DMatch>& vInlierMatches)
{
    vInlierMatches.clear();
    vInlierMatches.reserve(m12.size());
    double meanError = 0.0;
    Eigen::Matrix4d transformation4d = transformation4f.cast<double>();

    for (const auto& m : m12) {
        const cv::Point3f& origin = pF1->kps3Dc[m.queryIdx];
        const cv::Point3f& target = pF2->kps3Dc[m.trainIdx];

        if (origin.z == 0.0f || target.x == 0.0f)
            continue;

        double mahalDist = ErrorFunction2(Eigen::Vector4f(origin.x, origin.y, origin.z, 1.0), Eigen::Vector4f(target.x, target.y, target.z, 1.0), transformation4d);

        if (mahalDist > (mMaxMahalanobisDistance * mMaxMahalanobisDistance))
            continue;
        if (!(mahalDist >= 0.0))
            continue;

        meanError += mahalDist;
        vInlierMatches.push_back(m);
    }

    if (vInlierMatches.size() < 3) {
        meanError = 1e9;
    } else {
        meanError /= vInlierMatches.size();
        meanError = sqrt(meanError);
    }

    return meanError;
}

double Ransac::ErrorFunction2(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2, const Eigen::Matrix4d& transformation)
{
    static const double cam_angle_x = 58.0 / 180.0 * M_PI;
    static const double cam_angle_y = 45.0 / 180.0 * M_PI;
    static const double cam_resol_x = 640;
    static const double cam_resol_y = 480;
    static const double raster_stddev_x = 3 * tan(cam_angle_x / cam_resol_x);
    static const double raster_stddev_y = 3 * tan(cam_angle_y / cam_resol_y);
    static const double raster_cov_x = raster_stddev_x * raster_stddev_x;
    static const double raster_cov_y = raster_stddev_y * raster_stddev_y;
    static const bool use_error_shortcut = true;

    bool nan1 = std::isnan(x1(2));
    bool nan2 = std::isnan(x2(2));
    if (nan1 || nan2)
        return std::numeric_limits<double>::max();

    Eigen::Vector4d x_1 = x1.cast<double>();
    Eigen::Vector4d x_2 = x2.cast<double>();

    Eigen::Matrix4d tf_12 = transformation;
    Eigen::Vector3d mu_1 = x_1.head<3>();
    Eigen::Vector3d mu_2 = x_2.head<3>();
    Eigen::Vector3d mu_1_in_frame_2 = (tf_12 * x_1).head<3>(); // μ₁⁽²⁾  = T₁₂ μ₁⁽¹⁾

    if (use_error_shortcut) {
        double delta_sq_norm = (mu_1_in_frame_2 - mu_2).squaredNorm();
        double sigma_max_1 = std::max(raster_cov_x, DepthCovariance(mu_1(2)));
        double sigma_max_2 = std::max(raster_cov_x, DepthCovariance(mu_2(2)));

        if (delta_sq_norm > 2.0 * (sigma_max_1 + sigma_max_2))
            return std::numeric_limits<double>::max();
    }

    Eigen::Matrix3d rotation_mat = tf_12.block(0, 0, 3, 3);

    //Point 1
    Eigen::Matrix3d cov1 = Eigen::Matrix3d::Zero();
    cov1(0, 0) = raster_cov_x * mu_1(2);
    cov1(1, 1) = raster_cov_y * mu_1(2);
    cov1(2, 2) = DepthCovariance(mu_1(2));

    //Point2
    Eigen::Matrix3d cov2 = Eigen::Matrix3d::Zero();
    cov2(0, 0) = raster_cov_x * mu_2(2);
    cov2(1, 1) = raster_cov_y * mu_2(2);
    cov2(2, 2) = DepthCovariance(mu_2(2));

    Eigen::Matrix3d cov1_in_frame_2 = rotation_mat.transpose() * cov1 * rotation_mat;

    // Δμ⁽²⁾ =  μ₁⁽²⁾ - μ₂⁽²⁾
    Eigen::Vector3d delta_mu_in_frame_2 = mu_1_in_frame_2 - mu_2;
    if (std::isnan(delta_mu_in_frame_2(2)))
        return std::numeric_limits<double>::max();

    // Σc = (Σ₁ + Σ₂)
    Eigen::Matrix3d cov_mat_sum_in_frame_2 = cov1_in_frame_2 + cov2;
    //ΔμT Σc⁻¹Δμ
    double sqrd_mahalanobis_distance = delta_mu_in_frame_2.transpose() * cov_mat_sum_in_frame_2.llt().solve(delta_mu_in_frame_2);

    if (!(sqrd_mahalanobis_distance >= 0.0))
        return std::numeric_limits<double>::max();

    return sqrd_mahalanobis_distance;
}

double Ransac::DepthCovariance(double depth)
{
    static double stddev = DepthStdDev(depth);
    static double cov = stddev * stddev;
    return cov;
}

double Ransac::DepthStdDev(double depth)
{
    // From Khoselham and Elberink
    // Factor c for the standard deviation of depth measurements: sigma_Z = c * depth * depth.
    // Khoshelham 2012 (0.001425) seems to be a bit overconfident."
    static double depth_std_dev = 0.01;

    return depth_std_dev * depth * depth;
}

const std::vector<cv::DMatch>& Ransac::GetMatches() const { return mvMatches; }

Eigen::Matrix4f Ransac::GetTransformation() const { return mTransformation; }

float Ransac::GetRMSE() const { return rmse; }

void Ransac::SetMinInlierTh(uint minInlierTh) { mMinInlierTh = minInlierTh; }

void Ransac::SetIterations(int iters) { mIterations = iters; }

void Ransac::SetMaxMahalanobisDistance(float dist) { mMaxMahalanobisDistance = dist; }

void Ransac::SetSampleSize(uint sampleSize) { mSampleSize = sampleSize; }

uint Ransac::GetMinInlierTh() const { return mMinInlierTh; }

int Ransac::GetIterations() const { return mIterations; }

float Ransac::GetMaxMahalanobisDistance() const { return mMaxMahalanobisDistance; }

uint Ransac::GetSampleSize() const { return mSampleSize; }
