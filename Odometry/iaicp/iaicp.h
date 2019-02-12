#ifndef IAICP_H
#define IAICP_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Iaicp {
public:
    Iaicp();

    ~Iaicp();

    void SetInputSource(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pSource);

    void SetInputTarget(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pTarget);

    // Set up prediction of transformation
    void SetPredict(Eigen::Affine3f pred);

    // Performs the iterative registration
    void Run();

    // Returns the estimated transformation result
    Eigen::Affine3f GetTransResult() { return m_trans; }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr GetSalientSource() { return m_salientSrc; }

    void CheckAngles(Eigen::Matrix<float, 6, 1>& vec);

    Eigen::Affine3f toEigen(Eigen::Matrix<float, 6, 1> pose);

    Eigen::Matrix<float, 6, 1> toVector(Eigen::Affine3f pose);

private:
    // Performs one level of iterations of the IaICP method
    // maxDist: max. distance allowed between correspondences
    // offset: skipping pixel number,   refer l in the paper
    // maxiter: iteration number in this level
    void IterateLevel(float maxDist, int offset, int maxiter);

    void SampleSource();

    float ColorsImGray(pcl::PointXYZRGB a, pcl::PointXYZRGB b);

    float GetResidual(pcl::PointXYZRGB a, pcl::PointXYZRGB b);

    // Used for IterateLevel(), selected correspondences for each iteration
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt_;

    float intMedian, geoMedian, intMad, geoMad;

    // source and target frame
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_src;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_tgt;

    //salient points of the source frame
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_salientSrc;

    // Transformation that transforms source frame to target frame.
    Eigen::Affine3f m_trans;

    // Prediction of source2target transformaiton.
    Eigen::Affine3f m_predict;
};

#endif // IAICP_H
