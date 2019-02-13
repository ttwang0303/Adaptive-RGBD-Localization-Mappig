#include "pointclouddrawer.h"
#include "Core/frame.h"
#include "Core/landmark.h"
#include "Core/map.h"
#include "Utils/converter.h"
#include <boost/make_shared.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

PointCloudDrawer::PointCloudDrawer(Map* pMap)
    : mpMap(pMap)
{
    mpMapCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    mPointSize = 2.0f;
    mLineWidth = 0.9f;
    mLineWidthGraph = 1.5f;
}

void PointCloudDrawer::DrawPointCloud(bool drawLandmarks, bool drawDenseCloud, bool drawKFs)
{
    unique_lock<mutex> lock(mMutexPointCloud);

    // Draw Landmarks
    if (drawLandmarks)
        DrawLandmarks();

    vector<Frame*> vpKFs;
    if (drawDenseCloud || drawKFs) {
        vpKFs = mpMap->GetAllKeyFrames();
        if (drawKFs)
            sort(vpKFs.begin(), vpKFs.end(), [](const Frame* f1, const Frame* f2) { return f1->mnId < f2->mnId; });

        for (Frame* pKF : vpKFs) {
            // Draw dense cloud
            if (drawDenseCloud)
                DrawDenseCloud(pKF);

            // Draw pose
            if (drawKFs)
                DrawPoseKF(pKF);
        }

        // Draw graph
        if (drawKFs)
            DrawConnections(vpKFs);
    }
}

void PointCloudDrawer::DrawLandmarks()
{
    const vector<Landmark*> vpLandmarks = mpMap->GetAllLandmarks();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);

    for (size_t i = 0; i < vpLandmarks.size(); i += 2) {
        cv::Mat pos = vpLandmarks[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }

    glEnd();
}

void PointCloudDrawer::DrawDenseCloud(Frame* pKF)
{
    set<int>::iterator it = mKFids.find(pKF->mnId);

    // pKF has been included
    if (it != mKFids.end()) {
        size_t N = mpMapCloud->points.size();
        glPointSize(mPointSize - 0.5f);
        glBegin(GL_POINTS);

        for (size_t i = 0; i < N; ++i) {
            glColor3f(mpMapCloud->points[i].r / 255.0f, mpMapCloud->points[i].g / 255.0f, mpMapCloud->points[i].b / 255.0f);
            glVertex3f(mpMapCloud->points[i].x, mpMapCloud->points[i].y, mpMapCloud->points[i].z);
        }
        glEnd();
    }
    // pKF hasn't been included
    else {
        mKFids.insert(pKF->mnId);

        pKF->CreateCloud();
        pKF->VoxelGridFilterCloud(0.04f);
        pKF->StatisticalOutlierRemovalFilterCloud(50, 1.0);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapPointsCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud(*pKF->mpCloud, *mapPointsCloud, Converter::toMatrix4f(pKF->GetPose().inv()));

        for (int i = 0; i < mapPointsCloud->points.size(); ++i)
            mpMapCloud->points.push_back(mapPointsCloud->points[i]);

        size_t N = mpMapCloud->points.size();
        glPointSize(mPointSize - 0.5f);
        glBegin(GL_POINTS);

        for (size_t i = 0; i < N; ++i) {
            glColor3f(mpMapCloud->points[i].r / 255.0f, mpMapCloud->points[i].g / 255.0f, mpMapCloud->points[i].b / 255.0f);
            glVertex3f(mpMapCloud->points[i].x, mpMapCloud->points[i].y, mpMapCloud->points[i].z);
        }
        glEnd();
    }
}

void PointCloudDrawer::DrawPoseKF(Frame* pKF)
{
    static const float& w = 0.05;
    static const float h = w * 0.75;
    static const float z = w * 0.6;
    static const float r = 255.0f / 255.0f;
    static const float g = 153.0f / 255.0f;
    static const float b = 51.0f / 255.0f;

    cv::Mat Twc = pKF->GetPose().inv().t();
    glPushMatrix();
    glMultMatrixf(Twc.ptr<GLfloat>(0));
    glLineWidth(mLineWidthGraph);
    glColor3f(r, g, b);
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

    glEnd();
}

void PointCloudDrawer::DrawConnections(std::vector<Frame*>& vpOrderedKFs)
{
    glLineWidth(mLineWidthGraph);
    glColor3f(0.0f, 0.75f, 1.0f);
    glBegin(GL_LINES);
    for (size_t i = 0; i < vpOrderedKFs.size(); ++i) {
        if (i < vpOrderedKFs.size() - 1) {
            cv::Mat P = vpOrderedKFs[i]->GetCameraCenter();
            cv::Mat P1 = vpOrderedKFs[i + 1]->GetCameraCenter();

            glVertex3f(P.at<float>(0), P.at<float>(1), P.at<float>(2));
            glVertex3f(P1.at<float>(0), P1.at<float>(1), P1.at<float>(2));
        }
    }

    glEnd();
}
