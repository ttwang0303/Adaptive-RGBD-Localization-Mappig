#include "kabsch.h"
#include <Eigen/Eigen>

using namespace std;

template <typename T>
int sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}

Kabsch::Kabsch() {}

Eigen::Matrix4f Kabsch::Compute(const Eigen::MatrixXf &setA, const Eigen::MatrixXf &setB)
{
    mTransformation.setIdentity();
    if (setA.rows() == 0)
        return mTransformation;

    // Calculate center of mass
    Eigen::Vector3f centerOfMassA, centerOfMassB;
    for (int i = 0; i < 3; ++i) {
        centerOfMassA[i] = setA.col(i).mean();
        centerOfMassB[i] = setB.col(i).mean();
    }

    // Move center of mass
    Eigen::MatrixXf setAnew(setA.rows(), setA.cols());
    Eigen::MatrixXf setBnew(setB.rows(), setB.cols());
    for (int i = 0; i < setA.rows(); ++i) {
        setAnew.row(i) = setA.row(i) - centerOfMassA.transpose();
        setBnew.row(i) = setB.row(i) - centerOfMassB.transpose();
    }

    Eigen::MatrixXf A = setAnew.transpose() * setBnew;

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXf V = svd.matrixU();
    Eigen::MatrixXf W = svd.matrixV();

    Eigen::Matrix3f U = Eigen::MatrixXf::Identity(3, 3);
    U(2, 2) = sgn((A.determinant() != 0) ? A.determinant() : 1);

    // Optimal rotation matrix
    U = W * U * V.transpose();

    // Compute translation
    Eigen::Vector3f T;
    T = -centerOfMassA;
    T = U * T;
    T += centerOfMassB;

    // Optimal transformation
    mTransformation.matrix().block<3, 3>(0, 0) = U.block<3, 3>(0, 0);
    mTransformation.matrix().block<3, 1>(0, 3) = T.head<3>();
    return mTransformation;
}
