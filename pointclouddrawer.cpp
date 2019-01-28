#include "pointclouddrawer.h"
#include <boost/make_shared.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/io.h>

using namespace std;

PointCloudDrawer::PointCloudDrawer()
{
    mpSourceCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpTargetCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    mPointSize = 2.0f;
    mLineWidth = 0.9f;
}

void PointCloudDrawer::DrawPointCloud()
{
    unique_lock<mutex> lock(mMutexPointCloud);

    if (mpSourceCloud->points.empty() || mpTargetCloud->points.empty())
        return;
    if (mpSourceCloud->points.size() != mpTargetCloud->points.size())
        return;

    size_t N = mpSourceCloud->points.size();

    for (size_t i = 0; i < N; ++i) {
        glPointSize(mPointSize);
        glBegin(GL_POINTS);

        // Red for source
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(mpSourceCloud->points[i].x, mpSourceCloud->points[i].y, mpSourceCloud->points[i].z);

        // Blue for target
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(mpTargetCloud->points[i].x, mpTargetCloud->points[i].y, mpTargetCloud->points[i].z);

        glEnd();

        // Draw connecor
        glLineWidth(mLineWidth);
        glColor3f(0.0f, 0.75f, 0.0f);
        glBegin(GL_LINES);

        glVertex3f(mpSourceCloud->points[i].x, mpSourceCloud->points[i].y, mpSourceCloud->points[i].z);
        glVertex3f(mpTargetCloud->points[i].x, mpTargetCloud->points[i].y, mpTargetCloud->points[i].z);

        glEnd();
    }

    glEnd();
}

void PointCloudDrawer::AssignSourceCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pSource)
{
    unique_lock<mutex> lock(mMutexPointCloud);
    pcl::copyPointCloud(*pSource, *mpSourceCloud);
}

void PointCloudDrawer::AssignTargetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pTarget)
{
    unique_lock<mutex> lock(mMutexPointCloud);
    pcl::copyPointCloud(*pTarget, *mpTargetCloud);
}
