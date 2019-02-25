#include "pnpsolver.h"
#include "Core/frame.h"
#include "Core/landmark.h"
#include "Utils/common.h"
#include "Utils/converter.h"
#include <Eigen/StdVector>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

typedef g2o::VertexSE3Expmap Vertex;
typedef g2o::EdgeSE3ProjectXYZOnlyPose Edge;

int PnPSolver::Compute(Frame* pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // Set Frame vertex
    Vertex* vSE3 = new Vertex();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->GetPose()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set Landmark vertices
    const size_t N = pFrame->N;

    vector<Edge*> vpEdges;
    vector<size_t> vnIndexEdge;
    vpEdges.reserve(N);

    const double delta = sqrt(5.991);

    {
        unique_lock<mutex> lock(Landmark::mGlobalMutex);

        for (size_t i = 0; i < N; ++i) {
            Landmark* pLM = pFrame->GetLandmark(i);
            if (!pLM)
                continue;

            cv::Mat Xw = pLM->GetWorldPos();
            nInitialCorrespondences++;
            pFrame->SetInlier(i);

            Eigen::Matrix<double, 2, 1> obs;
            const cv::KeyPoint& kp = pFrame->mvKeys[i];
            obs << kp.pt.x, kp.pt.y;

            Edge* edge = new Edge();

            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            edge->setMeasurement(obs);
            const float sigma = 1.0f / (Xw.at<float>(2) * Xw.at<float>(2));
            edge->setInformation(Eigen::Matrix2d::Identity() * sigma);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            rk->setDelta(delta);

            edge->fx = static_cast<double>(Calibration::fx);
            edge->fy = static_cast<double>(Calibration::fy);
            edge->cx = static_cast<double>(Calibration::cx);
            edge->cy = static_cast<double>(Calibration::cy);

            edge->Xw[0] = Xw.at<float>(0);
            edge->Xw[1] = Xw.at<float>(1);
            edge->Xw[2] = Xw.at<float>(2);

            optimizer.addEdge(edge);

            vpEdges.push_back(edge);
            vnIndexEdge.push_back(i);
        }
    }

    if (nInitialCorrespondences < 3)
        return 0;

    const int its = 10;
    int nBad = 0;

    for (size_t it = 0; it < 4; it++) {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->GetPose()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its);

        nBad = 0;
        for (size_t i = 0; i < vpEdges.size(); i++) {
            Edge* edge = vpEdges[i];
            const size_t idx = vnIndexEdge[i];

            if (pFrame->IsOutlier(idx))
                edge->computeError();

            const double chi2 = edge->chi2();

            if (chi2 > 5.991) {
                pFrame->SetOutlier(idx);
                edge->setLevel(1);
                nBad++;
            } else {
                pFrame->SetInlier(idx);
                edge->setLevel(0);
            }

            if (it == 2)
                edge->setRobustKernel(nullptr);
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    Vertex* vSE3_recov = static_cast<Vertex*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences - nBad;
}
