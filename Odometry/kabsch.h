#ifndef KABSCH_H
#define KABSCH_H

#include <Eigen/Core>

class Kabsch {
public:
    Kabsch();

    Eigen::Matrix4f Compute(const Eigen::MatrixXf& setA, const Eigen::MatrixXf& setB);

private:
    Eigen::Matrix4f mTransformation;
};

#endif // KABSCH_H
