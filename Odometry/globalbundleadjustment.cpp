#include "globalbundleadjustment.h"
#include "Core/keyframe.h"
#include "Core/landmark.h"
#include "Core/map.h"
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

void GlobalBundleAdjustment::Compute(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<Landmark*> vpLMs = pMap->GetAllLandmarks();
    BundleAdjustment(vpKFs, vpLMs, nIterations, pbStopFlag, nLoopKF, bRobust);
}

void GlobalBundleAdjustment::BundleAdjustment(const std::vector<KeyFrame*>& vpKFs, const std::vector<Landmark*>& vpLMs, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedLM;
    vbNotIncludedLM.resize(vpLMs.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId == 0);
        optimizer.addVertex(vSE3);
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for (size_t i = 0; i < vpLMs.size(); i++) {
        Landmark* pLM = vpLMs[i];
        if (pLM->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pLM->GetWorldPos()));
        const int id = pLM->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*, size_t> observations = pLM->GetObservations();

        int nEdges = 0;
        // SET EDGES
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin();
             mit != observations.end(); mit++) {
            KeyFrame* pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint& kpUn = pKF->mvKeysUn[mit->second];

            if (pKF->mvuRight[mit->second] < 0) {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float& invSigma2 = 1;
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = static_cast<double>(Calibration::fx);
                e->fy = static_cast<double>(Calibration::fy);
                e->cx = static_cast<double>(Calibration::cx);
                e->cy = static_cast<double>(Calibration::cy);

                optimizer.addEdge(e);
            } else {
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float& invSigma2 = 1;
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = static_cast<double>(Calibration::fx);
                e->fy = static_cast<double>(Calibration::fy);
                e->cx = static_cast<double>(Calibration::cx);
                e->cy = static_cast<double>(Calibration::cy);
                e->bf = static_cast<double>(Calibration::mbf);

                optimizer.addEdge(e);
            }
        }

        if (nEdges == 0) {
            optimizer.removeVertex(vPoint);
            vbNotIncludedLM[i] = true;
        } else {
            vbNotIncludedLM[i] = false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    // Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == 0) {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        } else {
            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    // Points
    for (size_t i = 0; i < vpLMs.size(); i++) {
        if (vbNotIncludedLM[i])
            continue;

        Landmark* pLM = vpLMs[i];

        if (pLM->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pLM->mnId + maxKFid + 1));

        if (nLoopKF == 0) {
            pLM->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        } else {
            pLM->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pLM->mPosGBA);
            pLM->mnBAGlobalForKF = nLoopKF;
        }
    }
}
