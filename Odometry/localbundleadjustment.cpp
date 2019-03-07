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
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++) {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local Landmarks seen in Local KeyFrames
    list<Landmark*> lLocalLandmarks;
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        vector<Landmark*> vpLMs = (*lit)->GetLandmarks();
        for (vector<Landmark*>::iterator vit = vpLMs.begin(), vend = vpLMs.end(); vit != vend; vit++) {
            Landmark* pLM = *vit;
            if (pLM)
                if (!pLM->isBad())
                    if (pLM->mnBALocalForKF != pKF->mnId) {
                        lLocalLandmarks.push_back(pLM);
                        pLM->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for (list<Landmark*>::iterator lit = lLocalLandmarks.begin(), lend = lLocalLandmarks.end(); lit != lend; lit++) {
        map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
        for (map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
            KeyFrame* pKFi = mit->first;

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

    // Set Local KeyFrame vertices
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == 0);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for (list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalLandmarks.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<Landmark*> vpLandmarkEdgeMono;
    vpLandmarkEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<Landmark*> vpMapLandmarkStereo;
    vpMapLandmarkStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for (list<Landmark*>::iterator lit = lLocalLandmarks.begin(), lend = lLocalLandmarks.end(); lit != lend; lit++) {
        Landmark* pLM = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        cv::Mat Xw = pLM->GetWorldPos();
        vPoint->setEstimate(Converter::toVector3d(pLM->GetWorldPos()));
        int id = pLM->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*, size_t> observations = pLM->GetObservations();

        // Set edges
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
            KeyFrame* pKFi = mit->first;

            if (!pKFi->isBad()) {
                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if (pKFi->mvuRight[mit->second] < 0) {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float& sigma = 1.0f / (Xw.at<float>(2) * Xw.at<float>(2));
                    e->setInformation(Eigen::Matrix2d::Identity() * sigma);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = static_cast<double>(Calibration::fx);
                    e->fy = static_cast<double>(Calibration::fy);
                    e->cx = static_cast<double>(Calibration::cx);
                    e->cy = static_cast<double>(Calibration::cy);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpLandmarkEdgeMono.push_back(pLM);
                } else // Stereo observation
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float& sigma = 1.0f / (Xw.at<float>(2) * Xw.at<float>(2));
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * sigma;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = static_cast<double>(Calibration::fx);
                    e->fy = static_cast<double>(Calibration::fy);
                    e->cx = static_cast<double>(Calibration::cx);
                    e->cy = static_cast<double>(Calibration::cy);
                    e->bf = static_cast<double>(Calibration::mbf);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapLandmarkStereo.push_back(pLM);
                }
            }
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
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            Landmark* pLM = vpLandmarkEdgeMono[i];

            if (pLM->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive()) {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            Landmark* pLM = vpMapLandmarkStereo[i];

            if (pLM->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive()) {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame*, Landmark*>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        Landmark* pLM = vpLandmarkEdgeMono[i];

        if (pLM->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pLM));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        Landmark* pLM = vpMapLandmarkStereo[i];

        if (pLM->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pLM));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty()) {
        for (size_t i = 0; i < vToErase.size(); i++) {
            KeyFrame* pKFi = vToErase[i].first;
            Landmark* pMPi = vToErase[i].second;
            pKFi->EraseLandmark(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    // Keyframes
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    // Points
    for (list<Landmark*>::iterator lit = lLocalLandmarks.begin(), lend = lLocalLandmarks.end(); lit != lend; lit++) {
        Landmark* pLM = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(
            optimizer.vertex(pLM->mnId + maxKFid + 1));
        pLM->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    }
}
