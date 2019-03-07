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
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->GetPose()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const size_t N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(Landmark::mGlobalMutex);

        for (int i = 0; i < N; i++) {
            Landmark* pLM = pFrame->GetLandmark(i);
            if (pLM) {
                // Monocular observation
                if (pFrame->mvuRight[i] < 0) {
                    nInitialCorrespondences++;
                    pFrame->SetInlier(i);
                    cv::Mat Xw = pLM->GetWorldPos();

                    Eigen::Matrix<double, 2, 1> obs;
                    const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float sigma = 1.0f / (Xw.at<float>(2) * Xw.at<float>(2));
                    e->setInformation(Eigen::Matrix2d::Identity() * sigma);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = static_cast<double>(Calibration::fx);
                    e->fy = static_cast<double>(Calibration::fy);
                    e->cx = static_cast<double>(Calibration::cx);
                    e->cy = static_cast<double>(Calibration::cy);

                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else {
                    nInitialCorrespondences++;
                    pFrame->SetInlier(i);
                    cv::Mat Xw = pLM->GetWorldPos();

                    // SET EDGE
                    Eigen::Matrix<double, 3, 1> obs;
                    const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                    const float& kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float sigma = 1.0f / (Xw.at<float>(2) * Xw.at<float>(2));
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * sigma;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = static_cast<double>(Calibration::fx);
                    e->fy = static_cast<double>(Calibration::fy);
                    e->cx = static_cast<double>(Calibration::cx);
                    e->cy = static_cast<double>(Calibration::cy);
                    e->bf = static_cast<double>(Calibration::mbf);

                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
        }
    }

    if (nInitialCorrespondences < 3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation
    // as inlier/outlier At the next optimization, outliers are not included, but
    // at the end they can be classified as inliers again.
    const float chi2Mono[4] = { 5.991, 5.991, 5.991, 5.991 };
    const float chi2Stereo[4] = { 7.815, 7.815, 7.815, 7.815 };
    const int its[4] = { 10, 10, 10, 10 };

    int nBad = 0;
    for (size_t it = 0; it < 4; it++) {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->GetPose()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->IsOutlier(idx)) {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Mono[it]) {
                pFrame->SetOutlier(idx);
                e->setLevel(1);
                nBad++;
            } else {
                pFrame->SetInlier(idx);
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->IsOutlier(idx)) {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it]) {
                pFrame->SetOutlier(idx);
                e->setLevel(1);
                nBad++;
            } else {
                e->setLevel(0);
                pFrame->SetInlier(idx);
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences - nBad;
}
