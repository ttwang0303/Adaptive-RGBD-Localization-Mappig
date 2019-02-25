#include "localbundleadjustment.h"
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
#include <list>

using namespace std;

void LocalBundleAdjustment::Compute(KeyFrame* pKF, bool* pbStopFlag, Map* pMap)
{
    list<KeyFrame*> lLocalKFs;

    lLocalKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vpNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (KeyFrame* pKFneigh : vpNeighKFs) {
        pKFneigh->mnBALocalForKF = pKF->mnId;
        if (!pKFneigh->isBad())
            lLocalKFs.push_back(pKFneigh);
    }

    // Local Landmarks seen in Local KFs
    list<Landmark*> lLocalLandmarks;
    for (KeyFrame* pKFlocal : lLocalKFs) {
        vector<Landmark*> vpLMs = pKFlocal->GetLandmarks();
        for (Landmark* pLMlocal : vpLMs) {
            if (!pLMlocal)
                continue;
            if (pLMlocal->isBad())
                continue;

            if (pLMlocal->mnBALocalForKF != pKF->mnId) {
                lLocalLandmarks.push_back(pLMlocal);
                pLMlocal->mnBALocalForKF = pKF->mnId;
            }
        }
    }

    // Fixed KFs. KFs that see local Landmarks but that are not local KFs
    list<KeyFrame*> lFixedCameras;
    for (Landmark* pLMlocal : lLocalLandmarks) {
        map<KeyFrame*, size_t> observations = pLMlocal->GetObservations();
        for (auto& [pKFi, idx] : observations) {
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KF vertices
    for (KeyFrame* pKFlocal : lLocalKFs) {
        unsigned long KFid = pKFlocal->mnId;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFlocal->GetPose()));
        vSE3->setId(KFid);
        vSE3->setFixed(KFid == 0);
        optimizer.addVertex(vSE3);
        if (KFid > maxKFid)
            maxKFid = KFid;
    }

    // Set Fixed KF vertices
    for (KeyFrame* pKFfixed : lFixedCameras) {
        unsigned long KFid = pKFfixed->mnId;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFfixed->GetPose()));
        vSE3->setId(KFid);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (KFid > maxKFid)
            maxKFid = KFid;
    }

    // Set Landmark vertices
    const size_t nExpectedSize = (lLocalKFs.size() + lFixedCameras.size()) * lLocalLandmarks.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdges;
    vpEdges.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKF;
    vpEdgeKF.reserve(nExpectedSize);

    vector<Landmark*> vpLandmarkEdge;
    vpLandmarkEdge.reserve(nExpectedSize);

    const double thHuber = sqrt(5.991);

    for (Landmark* pLMlocal : lLocalLandmarks) {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        cv::Mat Xw = pLMlocal->GetWorldPos();
        vPoint->setEstimate(Converter::toVector3d(Xw));
        int id = pLMlocal->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*, size_t> observations = pLMlocal->GetObservations();

        // Set edges
        for (auto& [pKFobs, idxObs] : observations) {
            if (pKFobs->isBad())
                continue;

            const cv::KeyPoint& kp = pKFobs->mvKeys[idxObs];
            Eigen::Matrix<double, 2, 1> obs;
            obs << kp.pt.x, kp.pt.y;

            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFobs->mnId)));
            e->setMeasurement(obs);
            const float& sigma = 1.0f / (Xw.at<float>(2) * Xw.at<float>(2));
            e->setInformation(Eigen::Matrix2d::Identity() * sigma);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(thHuber);

            e->fx = static_cast<double>(Calibration::fx);
            e->fy = static_cast<double>(Calibration::fy);
            e->cx = static_cast<double>(Calibration::cx);
            e->cy = static_cast<double>(Calibration::cy);

            optimizer.addEdge(e);
            vpEdges.push_back(e);
            vpEdgeKF.push_back(pKFobs);
            vpLandmarkEdge.push_back(pLMlocal);
        }
    }

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore) {
        // Check inlier observations
        for (size_t i = 0; i < vpEdges.size(); ++i) {
            g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];
            Landmark* pLM = vpLandmarkEdge[i];

            if (pLM->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
                e->setLevel(1);

            e->setRobustKernel(nullptr);
        }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame*, Landmark*>> vToErase;
    vToErase.reserve(vpEdges.size());

    // Check inlier observations
    for (size_t i = 0; i < vpEdges.size(); ++i) {
        g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];
        Landmark* pLMi = vpLandmarkEdge[i];

        if (pLMi->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKF[i];
            vToErase.push_back({ pKFi, pLMi });
        }
    }

    // Get map mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty()) {
        for (size_t i = 0; i < vToErase.size(); ++i) {
            KeyFrame* pKFi = vToErase[i].first;
            Landmark* pLMi = vToErase[i].second;
            pKFi->EraseLandmark(pLMi);
            pLMi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    // KeyFrames
    for (KeyFrame* pKFlocal : lLocalKFs) {
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFlocal->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKFlocal->SetPose(Converter::toCvMat(SE3quat));
    }

    // Landmarks
    for (Landmark* pLMlocal : lLocalLandmarks) {
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pLMlocal->mnId + maxKFid + 1));
        pLMlocal->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    }
}
