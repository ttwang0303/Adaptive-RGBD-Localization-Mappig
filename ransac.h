#ifndef RANSAC_H
#define RANSAC_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>

class Frame;

class Ransac {
public:
    Ransac();

    Ransac(int iters, float maxMahalanobisDist, uint sampleSize);

    ~Ransac() {}

    // Compute the geometric relations T12 which allow us to estimate the motion
    // between the state pF1 and pF2 (pF1 -->[T12]--> pF2)
    bool Compute(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& m12);

    void SetIterations(int iters);
    void SetMaxMahalanobisDistance(float dist);
    void SetSampleSize(uint sampleSize);

    int GetIterations() const;
    float GetMaxMahalanobisDistance() const;
    uint GetSampleSize() const;

    const std::vector<cv::DMatch>& GetMatches() const;

    Eigen::Matrix4f GetTransformation() const;

    float GetRMSE() const;

private:
    std::vector<cv::DMatch> SampleMatches(const std::vector<cv::DMatch>& vMatches);

    Eigen::Matrix4f GetTransformFromMatches(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches, bool& valid);

    double ComputeInliersAndError(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& m12,
        const Eigen::Matrix4f& transformation4f, std::vector<cv::DMatch>& vInlierMatches);

    double ErrorFunction2(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2, const Eigen::Matrix4d& transformation);

    double DepthCovariance(double depth);

    double DepthStdDev(double depth);

    // Ransac parameters
    int mIterations;
    float mMaxMahalanobisDistance;
    uint mSampleSize;

    float rmse;
    std::vector<cv::DMatch> mvMatches;
    Eigen::Matrix4f mTransformation;
};

#endif // RANSAC_H
