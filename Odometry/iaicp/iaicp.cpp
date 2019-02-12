#include "iaicp.h"
#include "Utils/constants.h"
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>

using namespace std;

Iaicp::Iaicp()
{
    m_trans = Eigen::Affine3f::Identity();
    m_predict = Eigen::Affine3f::Identity();
}

Iaicp::~Iaicp() {}

void Iaicp::SetInputSource(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pSource)
{
    m_src.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    m_src = pSource;
    intMedian = 0.0f;
    intMad = 45.f;
}

void Iaicp::SetInputTarget(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pTarget)
{
    m_tgt.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    m_tgt = pTarget;

    geoMedian = 0.f;
    geoMad = 0.02f;
    intMedian = 0.f;
    intMad = 45.f;
}

void Iaicp::SetPredict(Eigen::Affine3f pred)
{
    m_predict = pred;
    m_trans = pred;
}

void Iaicp::Run()
{
    SampleSource();
    int iterPerLevel = 7;

    int offset = 7, maxDist = 0.15f;
    IterateLevel(maxDist, offset, iterPerLevel);

    offset = 3;
    maxDist = 0.06f;
    IterateLevel(maxDist, offset, iterPerLevel);

    offset = 1;
    maxDist = 0.02f;
    IterateLevel(maxDist, offset, 15);
}

void Iaicp::CheckAngles(Eigen::Matrix<float, 6, 1>& vec)
{
    for (size_t i = 3; i < 6; i++) {
        while (vec(i) > M_PI) {
            vec(i) -= 2 * M_PI;
        }
        while (vec(i) < -M_PI) {
            vec(i) += 2 * M_PI;
        }
    }
}

Eigen::Affine3f Iaicp::toEigen(Eigen::Matrix<float, 6, 1> pose)
{
    return pcl::getTransformation(pose(0), pose(1), pose(2), pose(3), pose(4), pose(5));
}

Eigen::Matrix<float, 6, 1> Iaicp::toVector(Eigen::Affine3f pose)
{
    Eigen::Matrix<float, 6, 1> temp;
    pcl::getTranslationAndEulerAngles(pose, temp(0), temp(1), temp(2), temp(3), temp(4), temp(5));
    CheckAngles(temp);

    return temp;
}

void Iaicp::IterateLevel(float maxDist, int offset, int maxiter)
{
    for (size_t iteration = 0; iteration < maxiter; iteration++) {
        tgt_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        src_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        std::vector<float> geoResiduals;
        std::vector<float> intResiduals;

        int counter = 0;
        for (size_t i = 0; i < m_salientSrc->points.size(); i++) {
            if (counter >= 150)
                break;

            int thisIndex = rand() % m_salientSrc->points.size();
            pcl::PointXYZRGB temp = m_salientSrc->points[thisIndex];
            pcl::PointXYZRGB pt = transformPoint(temp, m_trans);
            pcl::PointXYZRGB tgtpt;
            int xpos = int(floor(fx / pt.z * pt.x + cx));
            int ypos = int(floor(fy / pt.z * pt.y + cy));

            if (xpos >= (width) || ypos >= (height) || xpos < 0 || ypos < 0)
                continue;

            float maxWeight = 1e-10;
            int searchRange = 4;
            float intResidual, geoResidual;

            for (int xx = -searchRange; xx < searchRange + 1; xx++) {
                for (int yy = -searchRange; yy < searchRange + 1; yy++) {
                    float gridDist = sqrt(pow(float(xx), 2) + pow(float(yy), 2));
                    if (gridDist > (float)searchRange) {
                        continue;
                    }

                    int xpos_ = xpos + xx * (float)offset;
                    int ypos_ = ypos + yy * (float)offset;
                    if (xpos_ >= (width - 2) || ypos_ >= (height - 2) || xpos_ < 2 || ypos_ < 2)
                        continue;

                    pcl::PointXYZRGB pt2 = m_tgt->points[ypos_ * width + xpos_];
                    float dist = (pt.getVector3fMap() - pt2.getVector3fMap()).norm(); //geo. distance
                    if (dist == dist) {
                        float residual = GetResidual(pt2, pt);

                        if (residual == residual) {
                            float geoWeight = 1e2f * (6.f / (5.f + pow((dist) / (geoMad), 2)));
                            float colWeight = 1e2f * (6.f / (5.f + pow((residual - intMedian) / intMad, 2)));
                            float thisweight = geoWeight * colWeight;

                            if (thisweight == thisweight && thisweight > maxWeight) {
                                tgtpt = pt2;
                                maxWeight = thisweight;
                                intResidual = residual;
                                geoResidual = dist;
                            }
                        }
                    }
                }
            }

            if (maxWeight > 0) {
                if ((m_salientSrc->points[thisIndex].getVector3fMap() - tgtpt.getVector3fMap()).norm() < 1000.f) {
                    src_->points.push_back(pt);
                    tgt_->points.push_back(tgtpt);

                    intResidual = GetResidual(tgtpt, pt);
                    geoResidual = (pt.getVector3fMap() - tgtpt.getVector3fMap()).norm();
                    intResiduals.push_back(intResidual);
                    geoResiduals.push_back(geoResidual);
                    counter++;
                }
            }
        }

        // Estimate median and deviation for both intensity and geometry residuals
        vector<float> temp = geoResiduals;
        sort(temp.begin(), temp.end());
        geoMedian = temp[temp.size() - temp.size() / 2];
        for (size_t i = 0; i < temp.size(); i++)
            temp[i] = fabs(temp[i] - geoMedian);

        sort(temp.begin(), temp.end());
        geoMad = 1.f * 1.4826 * temp[temp.size() / 2] + 1e-11;
        for (size_t i = 0; i < geoResiduals.size(); i++)
            geoResiduals[i] = (6.f / (5.f + pow((geoResiduals[i]) / geoMad, 2)));

        temp.clear();
        temp = intResiduals;
        sort(temp.begin(), temp.end());
        intMedian = temp[temp.size() - temp.size() / 2];
        for (size_t i = 0; i < temp.size(); i++)
            temp[i] = fabs(temp[i] - intMedian);

        sort(temp.begin(), temp.end());
        intMad = 1.f * 1.4826 * temp[temp.size() / 2] + 1e-11;
        for (size_t i = 0; i < intResiduals.size(); i++)
            intResiduals[i] = (6.f / (5.f + pow((intResiduals[i] - intMedian) / intMad, 2)));

        pcl::TransformationFromCorrespondences transFromCorr;
        for (size_t i = 0; i < src_->points.size(); i++) {
            Eigen::Vector3f from(src_->points.at(i).x, src_->points.at(i).y, src_->points.at(i).z);
            Eigen::Vector3f to(tgt_->points.at(i).x, tgt_->points.at(i).y, tgt_->points.at(i).z);
            float sensorRel = 1.f / (0.0012 + 0.0019 * pow(src_->points.at(i).z - 0.4, 2));
            transFromCorr.add(from, to, geoResiduals[i] * intResiduals[i] * sensorRel);
        }

        Eigen::Affine3f increTrans = transFromCorr.getTransformation();
        m_trans = toEigen(toVector(increTrans * m_trans));
    }
}

void Iaicp::SampleSource()
{
    m_salientSrc.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    int cnt = 0;
    int begin = 2 + rand() % 2;

    for (size_t i = begin; i < width - begin - 4; i += 2) {
        for (size_t j = begin; j < height - begin - 4; j += 2) {
            pcl::PointXYZRGB pt = m_src->points[j * width + i];
            if (pt.z != pt.z || pt.z > 8.f)
                continue;

            // Warp to target image
            pcl::PointXYZRGB ptwarp = pt;
            ptwarp = pcl::transformPoint(ptwarp, m_predict);
            int xpos = int(floor(fx / ptwarp.z * ptwarp.x + cx));
            int ypos = int(floor(fy / ptwarp.z * ptwarp.y + cy));
            if (xpos >= width - 3 || ypos >= height - 3 || xpos < 3 || ypos < 3)
                continue;

            // Check whether backgfloor point
            float z_ = m_src->points[j * width + i].z;
            float diff1 = z_ - m_src->points[j * width + i + 2].z;
            float diff2 = z_ - m_src->points[j * width + i - 2].z;
            float diff3 = z_ - m_src->points[(j - 2) * width + i].z;
            float diff4 = z_ - m_src->points[(j + 2) * width + i].z;
            if (diff1 != diff1 || diff2 != diff2 || diff3 != diff3 || diff4 != diff4)
                continue;

            float thres = 0.021 * z_;
            if (diff1 > thres || diff2 > thres || diff3 > thres || diff4 > thres)
                continue;

            // Image gradient
            float sim1 = ColorsImGray(m_src->points[j * width + i - 4], m_src->points[j * width + i + 4]);
            float sim2 = ColorsImGray(m_src->points[(j - 4) * width + i], m_src->points[(j + 4) * width + i]);
            if ((sim1 == sim1 && sim1 <= 0.85f) || (sim2 == sim2 && sim2 <= 0.85f)) {
                m_salientSrc->points.push_back(m_src->points[j * width + i]);
                cnt++;
                continue;
            }

            // Intensity residual
            float residual = fabs(GetResidual(m_tgt->points[ypos * width + xpos], pt));
            if (fabs(residual) > 100.f) {
                m_salientSrc->points.push_back(m_src->points[j * width + i]);
                cnt++;
                continue;
            }

            // Depth gradient
            if (fabs(diff1 - diff2) > 0.03f * z_ || fabs(diff3 - diff4) > 0.03f * z_) {
                m_salientSrc->points.push_back(m_src->points[j * width + i]);
                cnt++;
                continue;
            }
        }
    }

    if (cnt < 200) {
        for (size_t i = 0; i < 1000; i++)
            m_salientSrc->points.push_back(m_src->points[rand() % m_src->points.size()]);
    }

    vector<int> indices;
    pcl::removeNaNFromPointCloud(*m_salientSrc, *m_salientSrc, indices);
}

float Iaicp::ColorsImGray(pcl::PointXYZRGB a, pcl::PointXYZRGB b)
{
    float r1, g1, b1, r2, g2, b2;
    r1 = float(a.r);
    r2 = float(b.r);
    g1 = float(a.g);
    g2 = float(b.g);
    b1 = float(a.b);
    b2 = float(b.b);
    return 1.f - fabs(0.299f * r1 + 0.587f * g1 + 0.114f * b1 - 0.299f * r2 - 0.587f * g2 - 0.114f * b2) / 255.f;
}

float Iaicp::GetResidual(pcl::PointXYZRGB a, pcl::PointXYZRGB b)
{
    float r1, g1, b1, r2, g2, b2;
    r1 = float(a.r);
    r2 = float(b.r);
    g1 = float(a.g);
    g2 = float(b.g);
    b1 = float(a.b);
    b2 = float(b.b);
    return (0.299f * r1 + 0.587f * g1 + 0.114f * b1 - 0.299f * r2 - 0.587f * g2 - 0.114f * b2);
}
