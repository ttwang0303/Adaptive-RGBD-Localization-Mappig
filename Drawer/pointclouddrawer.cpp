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
    //    mpSourceCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    //    mpTargetCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpMapCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    mPointSize = 2.0f;
    mLineWidth = 0.9f;
}

void PointCloudDrawer::DrawPointCloud()
{
    unique_lock<mutex> lock(mMutexPointCloud);

    //    if (!mpSourceCloud->points.empty() && !mpTargetCloud->points.empty()) {
    //        size_t N = mpSourceCloud->points.size();

    //        for (size_t i = 0; i < N; ++i) {
    //            glPointSize(mPointSize);
    //            glBegin(GL_POINTS);

    //            // Red for source
    //            glColor3f(1.0f, 0.0f, 0.0f);
    //            glVertex3f(mpSourceCloud->points[i].x, mpSourceCloud->points[i].y, mpSourceCloud->points[i].z);

    //            // Blue for target
    //            glColor3f(0.0f, 0.0f, 1.0f);
    //            glVertex3f(mpTargetCloud->points[i].x, mpTargetCloud->points[i].y, mpTargetCloud->points[i].z);

    //            glEnd();

    //            // Draw connector
    //            glLineWidth(mLineWidth);
    //            glColor3f(0.0f, 0.75f, 0.0f);
    //            glBegin(GL_LINES);

    //            glVertex3f(mpSourceCloud->points[i].x, mpSourceCloud->points[i].y, mpSourceCloud->points[i].z);
    //            glVertex3f(mpTargetCloud->points[i].x, mpTargetCloud->points[i].y, mpTargetCloud->points[i].z);

    //            glEnd();
    //        }

    //        glEnd();
    //    }

    const float& w = 0.05;
    const float h = w * 0.75;
    const float z = w * 0.6;

    // Draw Landmarks
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);

    const vector<Landmark*> vpLandmarks = mpMap->GetAllLandmarks();
    vector<Frame*> vpKFs = mpMap->GetAllKeyFrames();

    sort(vpKFs.begin(), vpKFs.end(), [](const Frame* f1, const Frame* f2) {
        return f1->mnId < f2->mnId;
    });

    for (size_t i = 0; i < vpLandmarks.size(); i += 2) {
        cv::Mat pos = vpLandmarks[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }

    glEnd();

    //    // Draw dense cloud
    for (Frame* pKF : vpKFs) {
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

        // Draw poses
        cv::Mat Twc = pKF->GetPose().inv().t();
        glPushMatrix();
        glMultMatrixf(Twc.ptr<GLfloat>(0));
        glLineWidth(1.5);
        glColor3f(255.0f / 255.0f, 153.0f / 255.0f, 51.0f / 255.0f);
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

    glLineWidth(1.5f);
    glColor3f(0.0f, 0.75f, 1.0f);
    glBegin(GL_LINES);
    for (size_t i = 0; i < vpKFs.size(); ++i) {
        if (i < vpKFs.size() - 1) {
            cv::Mat P = vpKFs[i]->GetCameraCenter();
            cv::Mat P1 = vpKFs[i + 1]->GetCameraCenter();

            glVertex3f(P.at<float>(0), P.at<float>(1), P.at<float>(2));
            glVertex3f(P1.at<float>(0), P1.at<float>(1), P1.at<float>(2));
        }
    }
    glEnd();

    //    if (!mpMapCloud->points.empty()) {
    //        size_t N = mpMapCloud->points.size();

    //        glPointSize(mPointSize - 1.0f);
    //        glBegin(GL_POINTS);

    //        for (size_t i = 0; i < N; ++i) {
    //            glColor3f(mpMapCloud->points[i].r / 255.0f, mpMapCloud->points[i].g / 255.0f, mpMapCloud->points[i].b / 255.0f);
    //            glVertex3f(mpMapCloud->points[i].x, mpMapCloud->points[i].y, mpMapCloud->points[i].z);
    //        }

    //        glEnd();

    //        // Draw poses
    //        for (int i = 1; i < mvPoses.size(); ++i) {
    //            glPointSize(mPointSize * 4);
    //            glBegin(GL_POINTS);
    //            glColor3f(r, g, b);
    //            glVertex3f(mvPoses[i].at<float>(0, 3), mvPoses[i].at<float>(1, 3), mvPoses[i].at<float>(2, 3));
    //            glEnd();

    //            if (i < mvPoses.size() - 1) {
    //                glLineWidth(4);
    //                glBegin(GL_LINES);

    //                glVertex3f(mvPoses[i].at<float>(0, 3), mvPoses[i].at<float>(1, 3), mvPoses[i].at<float>(2, 3));
    //                glVertex3f(mvPoses[i + 1].at<float>(0, 3), mvPoses[i + 1].at<float>(1, 3), mvPoses[i + 1].at<float>(2, 3));

    //                glEnd();
    //            }
    //        }
    //    }

    //    glEnd();
}

//void PointCloudDrawer::UpdateSourceCloud(pcl::PointCloud<pcl::PointXYZ>& pSource)
//{
//    unique_lock<mutex> lock(mMutexPointCloud);
//    pcl::copyPointCloud(pSource, *mpSourceCloud);
//}

//void PointCloudDrawer::UpdateTargetCloud(pcl::PointCloud<pcl::PointXYZ>& pTarget)
//{
//    unique_lock<mutex> lock(mMutexPointCloud);
//    pcl::copyPointCloud(pTarget, *mpTargetCloud);
//}

//void PointCloudDrawer::UpdateMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pMapCloud, const cv::Mat& pose)
//{
//    unique_lock<mutex> lock(mMutexPointCloud);

//    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
//    float resolution = 0.03f;
//    voxel.setLeafSize(resolution, resolution, resolution);
//    voxel.setInputCloud(pMapCloud);
//    voxel.filter(*pMapCloud);

//    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
//    sor.setInputCloud(pMapCloud);
//    sor.setMeanK(50);
//    sor.setStddevMulThresh(1.0);
//    sor.filter(*pMapCloud);

//    for (int i = 0; i < pMapCloud->points.size(); ++i)
//        mpMapCloud->points.push_back(pMapCloud->points[i]);

//    mvPoses.push_back(pose.clone());

//    r = (double)rand() / (double)RAND_MAX;
//    g = (double)rand() / (double)RAND_MAX;
//    b = (double)rand() / (double)RAND_MAX;
//}
