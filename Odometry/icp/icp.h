#ifndef ICP_H
#define ICP_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "kdtree.h"
#include "matrix.h"

class Icp {

public:
    // constructor
    // input: M ....... pointer to first model point
    //        M_num ... number of model points
    //        dim   ... dimensionality of model points (2 or 3)
    Icp(double* M, const int32_t M_num, const int32_t dim);

    // deconstructor
    virtual ~Icp();

    // set maximum number of iterations (1. stopping criterion)
    void setMaxIterations(int32_t val) { m_max_iter = val; }

    // set minimum delta of rot/trans parameters (2. stopping criterion)
    void setMinDeltaParam(double val) { m_min_delta = val; }

    double getInlierRatio()
    {
        return m_inlier_ratio;
    }

    double getInlierCount()
    {
        return m_active.size();
    }

    // fit template to model yielding R,t (M = R*T + t)
    // input:  T ....... pointer to first template point
    //         T_num ... number of template points
    //         R ....... initial rotation matrix
    //         t ....... initial translation vector
    //         indist .. inlier distance (if <=0: use all points)
    // output: R ....... final rotation matrix
    //         t ....... final translation vector
    double fit(double* T, const int32_t T_num, Matrix& R, Matrix& t, double indist = -1);

private:
    // iterative fitting
    void fitIterate(double* T, const int32_t T_num, Matrix& R, Matrix& t, double indist = -1);

    // inherited classes need to overwrite these functions
    virtual double fitStep(double* T, const int32_t T_num, Matrix& R, Matrix& t, const std::vector<int32_t>& active) = 0;
    virtual std::vector<int32_t> getInliers(double* T, const int32_t T_num, const Matrix& R, const Matrix& t, const double indist) = 0;

    virtual double getResidual(double* T, const int32_t T_num, const Matrix& R, const Matrix& t, const std::vector<int>& active) = 0;

protected:
    // kd tree of model points
    kdtree::KDTree* m_kd_tree;
    kdtree::KDTreeArray m_kd_data;

    int32_t m_dim; // dimensionality of model + template data (2 or 3)
    int32_t m_max_iter; // max number of iterations
    double m_min_delta; // min parameter delta

    std::vector<int> m_active; //inliers

    double m_inlier_ratio; // active.size()/ T_num
    double m_residual; // residual of icp
    Matrix m_covariance; // covariance of the result
};

#endif // ICP_H
