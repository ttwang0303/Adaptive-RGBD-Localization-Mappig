#include "mapdrawer.h"
#include "Core/keyframe.h"
#include "Core/landmark.h"
#include "Core/map.h"
#include "Utils/converter.h"
#include <boost/make_shared.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

MapDrawer::MapDrawer(Map* pMap)
    : mpMap(pMap)
{
    mpMapCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    mPointSize = 2.0f;
    mKeyFrameSize = 0.05f;
    mKeyFrameLineWidth = 1;
    mGraphLineWidth = 0.9f;
}

size_t MapDrawer::DrawLandmarks()
{
    const vector<Landmark*> vpLandmarks = mpMap->GetAllLandmarks();
    if (vpLandmarks.empty())
        return 0;

    size_t nLMs = 0;
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);

    for (size_t i = 0; i < vpLandmarks.size(); i++) {
        Landmark* pLM = vpLandmarks[i];
        if (!pLM)
            continue;
        if (pLM->isBad())
            continue;

        cv::Mat pos = pLM->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        nLMs++;
    }

    glEnd();
    return nLMs;
}

size_t MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    static const float& w = mKeyFrameSize;
    static const float h = mKeyFrameSize * 0.75f;
    static const float z = mKeyFrameSize * 0.6f;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
        return 0;

    sort(vpKFs.begin(), vpKFs.end(), [](const KeyFrame* pKF1, const KeyFrame* pKF2) {
        return pKF1->mnId < pKF2->mnId;
    });

    if (bDrawKF) {
        KeyFrame* pLastKF = vpKFs.back();
        cv::Mat Ow = pLastKF->GetCameraCenter();
        vector<Landmark*> vpLMs = pLastKF->GetLandmarks();

        glLineWidth(mKeyFrameLineWidth);
        glColor4f(0.82f, 0.82f, 0.82f, 0.35f);
        glBegin(GL_LINES);

        for (size_t i = 0; i < pLastKF->N; i++) {
            Landmark* pLM = vpLMs[i];
            if (!pLM)
                continue;
            if (pLastKF->IsOutlier(i))
                continue;

            cv::Mat Xw = pLM->GetWorldPos();

            glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
            glVertex3f(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2));
        }
        glEnd();

        for (KeyFrame* pKF : vpKFs) {
            cv::Mat Twc = pKF->GetPoseInv().t();

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(w, h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, h, z);

            glVertex3f(w, h, z);
            glVertex3f(w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(-w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(w, h, z);

            glVertex3f(-w, -h, z);
            glVertex3f(w, -h, z);
            glEnd();

            glPopMatrix();
        }
    }

    if (bDrawGraph) {
        glLineWidth(mGraphLineWidth);
        glColor3f(0.0f, 0.75f, 1.0f);
        glBegin(GL_LINES);

        for (KeyFrame* pKF : vpKFs) {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = pKF->GetCovisiblesByWeight(100);
            cv::Mat Ow = pKF->GetCameraCenter();

            if (!vCovKFs.empty()) {
                for (KeyFrame* pKFcov : vCovKFs) {
                    if (pKFcov->mnId < pKF->mnId)
                        continue;
                    cv::Mat Ow2 = pKFcov->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = pKF->GetParent();
            if (pParent) {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
            }
        }

        glEnd();
    }

    return vpKFs.size();
}
