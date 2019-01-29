#include "pointtopointicp.h"
#include "Core/frame.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/icp/types_icp.h>

using namespace std;

PointToPointICP::PointToPointICP()
{
}

Eigen::Matrix4f PointToPointICP::Compute(const Frame* srcFrame, const Frame* tgtFrame, const std::vector<cv::DMatch>& vMatches, Eigen::Matrix4f& guess)
{
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    g2o::BlockSolverX::LinearSolverType* pLinearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* pSolver = new g2o::BlockSolverX(pLinearSolver);
    g2o::OptimizationAlgorithmLevenberg* pSolverAlgorithm = new g2o::OptimizationAlgorithmLevenberg(pSolver);

    optimizer.setAlgorithm(pSolverAlgorithm);

    g2o::VertexSE3* tgtVertex = new g2o::VertexSE3();
    tgtVertex->setId(0);
    tgtVertex->setFixed(true);

    g2o::VertexSE3* srcVertex = new g2o::VertexSE3();
    srcVertex->setId(1);

    Eigen::Matrix3f R = guess.block(0, 0, 3, 3);
    Eigen::Isometry3f est = Eigen::Isometry3f::Identity();
    Eigen::AngleAxisf angle(R);
    est = angle;
    est(0, 3) = guess(0, 3);
    est(1, 3) = guess(1, 3);
    est(2, 3) = guess(2, 3);

    srcVertex->setEstimate(est.cast<double>());

    optimizer.addVertex(tgtVertex);
    optimizer.addVertex(srcVertex);

    for (const auto& m : vMatches) {
        g2o::Edge_V_V_GICP* edge = new g2o::Edge_V_V_GICP();
        edge->setVertex(0, tgtVertex);
        edge->setVertex(1, srcVertex);

        cv::Point3f srcPc = srcFrame->mvKps3Dc[m.queryIdx];
        cv::Point3f tgtPc = tgtFrame->mvKps3Dc[m.trainIdx];

        g2o::EdgeGICP measurement;
        measurement.pos0 = Eigen::Vector3d(tgtPc.x, tgtPc.y, tgtPc.z);
        measurement.pos1 = Eigen::Vector3d(srcPc.x, srcPc.y, srcPc.z);

        edge->setMeasurement(measurement);
        edge->information().setIdentity();
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
        rk->setDelta(0.08);
        edge->setRobustKernel(rk);

        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    Eigen::Matrix4f T = srcVertex->estimate().matrix().cast<float>();
    return T;
}
