 pcl::visualization::PCLVisualizer* p;
int vp1, vp2;


 
 void PairAlign(const PointCloud::Ptr srcCloud, const PointCloud::Ptr tgtCloud, PointCloud::Ptr outCloud, cv::Mat& finalTransform, bool downsample)
{
    PointCloud::Ptr src(new PointCloud);
    PointCloud::Ptr tgt(new PointCloud);
    pcl::VoxelGrid<PointT> grid;

    if (downsample) {
        float leaf = 0.02f;
        grid.setLeafSize(leaf, leaf, leaf);
        grid.setInputCloud(srcCloud);
        grid.filter(*src);

        grid.setInputCloud(tgtCloud);
        grid.filter(*tgt);
    } else {
        src = srcCloud;
        tgt = tgtCloud;
    }

    // Align
    pcl::IterativeClosestPoint<PointT, PointT> reg;
    reg.setTransformationEpsilon(1e-9);
    reg.setMaxCorrespondenceDistance(0.03);
    reg.setMaximumIterations(20);

    reg.setInputSource(src);
    reg.setInputTarget(tgt);

    Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity() /*, prev, targetToSource*/;
    PointCloud::Ptr regResult(new PointCloud);

    reg.align(*regResult);

    if (reg.hasConverged())
        Ti = reg.getFinalTransformation();

    //    for (int i = 0; i < 10; ++i) {
    //        src = regResult;

    //        // Estimate
    //        reg.setInputSource(src);
    //        reg.align(*regResult);

    //        Ti = reg.getFinalTransformation() * Ti;

    //        if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
    //            reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);

    //        prev = reg.getLastIncrementalTransformation();

    //        ShowCloudsRight(tgt, src);
    //    }

    // Get the transformation from target to source
    //    targetToSource = Ti.inverse();

    // Transform target back in source frame
    //    pcl::transformPointCloud(*tgtCloud, *outCloud, targetToSource);

    //    p->removePointCloud("source");
    //    p->removePointCloud("target");
    //    p->addPointCloud(outCloud, "target", vp2);
    //    p->addPointCloud(srcCloud, "source", vp2);
    //    p->spinOnce(1);

    //    p->removePointCloud("source");
    //    p->removePointCloud("target");

    //    *outCloud += *srcCloud;
    finalTransform = Converter::toCvMat(Ti);
}

void ShowCloudsLeft(const PointCloud::Ptr srcCloud, const PointCloud::Ptr tgtCloud)
{
    p->removePointCloud("vp1_target");
    p->removePointCloud("vp1_source");

    p->addPointCloud(tgtCloud, "vp1_target", vp1);
    p->addPointCloud(srcCloud, "vp1_source", vp1);

    p->spinOnce(1);
}

void ShowCloudsRight(const PointCloud::Ptr tgtCloud, const PointCloud::Ptr srcCloud)
{
    p->removePointCloud("source");
    p->removePointCloud("target");

    p->addPointCloud(tgtCloud, "target", vp2);
    p->addPointCloud(srcCloud, "source", vp2);

    p->spinOnce(1);
}
 
 cv::Mat RansacCv(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& m12)
{
    vector<cv::Point3f> vObjectPoints;
    vector<cv::Point2f> vImagePoints;
    vector<cv::DMatch> vValidMatches;

    for (const auto& m : m12) {
        if (pF1->kps3Dc[m.queryIdx].z > 0 && pF2->kps3Dc[m.trainIdx].z > 0) {
            cv::Point3f obj = pF1->kps3Dc[m.queryIdx];
            cv::Point2f img = pF2->kps[m.trainIdx].pt;
            vValidMatches.push_back(m);

            vObjectPoints.push_back(obj);
            vImagePoints.push_back(img);
        }
    }

    if (vObjectPoints.size() < 15)
        return cv::Mat();

    cv::Mat r, t, inliers, Rt;
    bool bOK = false;

    try {
        double camera_matrix_data[3][3] = {
            { fx, 0, cx },
            { 0, fy, cy },
            { 0, 0, 1 }
        };

        cv::Mat mK(3, 3, CV_64F, camera_matrix_data);

        bOK = cv::solvePnPRansac(vObjectPoints, vImagePoints, mK, cv::Mat(), r, t, false, 500, 2.0f, 0.85, inliers);
        Rt = Converter::toHomogeneous(r, t);

        bool c1 = !bOK;
        bool c2 = cv::norm(Rt) > 100.0;
        bool c3 = inliers.rows < 15;

        if (c1 || c2 || c3)
            bOK = false;
    } catch (cv::Exception& e) {
        bOK = false;
        cout << "Ransac Fails " << e.what() << endl;
    }

    if (!bOK)
        return cv::Mat();

    // Discard outliers
    m12.clear();
    m12.reserve(inliers.rows);
    for (int i = 0; i < inliers.rows; ++i) {
        int n = inliers.at<int>(i);
        m12.push_back(vValidMatches[n]);
    }

    return Rt;
}

cv::Mat Ransac2(Frame* pF1, Frame* pF2, vector<cv::DMatch>& m12)
{
    vector<cv::Point2f> p1, p2;
    for (int i = 0; i < m12.size(); ++i) {
        p1.push_back(pF1->kps[m12[i].queryIdx].pt);
        p2.push_back(pF2->kps[m12[i].trainIdx].pt);
    }

    vector<uchar> status;
    cv::Mat F = cv::findFundamentalMat(p1, p2, status, cv::FM_RANSAC, 2.0, 0.85);

    vector<cv::DMatch> vInliers;
    for (int j = 0; j < status.size(); ++j) {
        if (int(status[j]) != 0) {
            vInliers.push_back(m12[j]);
        }
    }

    m12 = vInliers;

    return F;
}

void Ransac3D(Frame* pF1, Frame* pF2, std::vector<cv::DMatch>& m12, pcl::CorrespondencesPtr inliers, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2)
{
    // Generate the matches pointcloud
    for (const auto& m : m12) {
        const cv::Point3f& pQ = pF1->kps3Dc[m.queryIdx];
        const cv::Point3f& pT = pF2->kps3Dc[m.trainIdx];

        if (pQ.z <= 0 || pT.z <= 0)
            continue;

        pcl::PointXYZ srcP(pQ.x, pQ.y, pQ.z);
        cloud1->points.push_back(srcP);

        pcl::PointXYZ tgtP(pT.x, pT.y, pT.z);
        cloud2->points.push_back(tgtP);
    }

    cloud1->height = 1;
    cloud1->width = cloud1->points.size();
    cloud1->is_dense = false;

    cloud2->height = 1;
    cloud2->width = cloud2->points.size();
    cloud2->is_dense = false;

    // Ransac use keypoint cloud
    pcl::CorrespondencesPtr correspondenceIn(new pcl::Correspondences);

    for (int i = 0; i < cloud1->size(); ++i) {
        pcl::Correspondence corrIn;
        corrIn.index_query = i;
        corrIn.index_match = i;
        corrIn.weight = 1.0 / (cloud1->points[i].z * cloud2->points[i].z);
        correspondenceIn->push_back(corrIn);
    }

    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>::Ptr
        ransac(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>);
    ransac->setInputSource(cloud1);
    ransac->setInputTarget(cloud2);
    ransac->setInlierThreshold(0.05);
    ransac->setMaximumIterations(500);
    ransac->getRemainingCorrespondences(*correspondenceIn, *inliers);

    //    Eigen::Matrix4f initialT = ransac->getBestTransformation();
    //    cv::Mat initT = Converter::toCvMat(initialT);
    //    return initT;
}

//bool GetRelativeTransformation(Frame* pF1, Frame* pF2, vector<cv::DMatch>& initialMatches12, float& rmse, vector<cv::DMatch>& matches, Eigen::Matrix4f& resultingTransformation)
//{
//    if (initialMatches12.size() <= 20)
//        return false;

//    uint minInlierThreshold = 20;
//    if (minInlierThreshold > 0.75 * initialMatches12.size())
//        minInlierThreshold = 0.75 * initialMatches12.size();

//    double inlierError;
//    const float maxDistM = 3;
//    const int ransacIts = 200;
//    matches.clear();
//    resultingTransformation = Eigen::Matrix4f::Identity();
//    rmse = 1e6;
//    uint validIts = 0;
//    const uint sampleSize = 4;
//    bool validTf = false;

//    sort(initialMatches12.begin(), initialMatches12.end());

//    int realIts = 0;
//    for (int n = 0; (n < ransacIts && initialMatches12.size() >= sampleSize); n++) {
//        double refinedError = 1e6;
//        vector<cv::DMatch> refinedMatches;
//        vector<cv::DMatch> inlier = SampleMatches(sampleSize, initialMatches12);
//        Eigen::Matrix4f refinedTransformation = Eigen::Matrix4f::Identity();

//        realIts++;
//        for (int refinements = 1; refinements < 20; refinements++) {
//            Eigen::Matrix4f transformation = GetTransformFromMatches(pF1, pF2, inlier, validTf, maxDistM);

//            if (!validTf || transformation != transformation)
//                break;

//            ComputeInliersAndError(pF1, pF2, initialMatches12, transformation,
//                std::max(minInlierThreshold, static_cast<uint>(refinedMatches.size())),
//                inlier, inlierError, maxDistM * maxDistM);

//            if (inlier.size() < minInlierThreshold || inlierError > maxDistM)
//                break;

//            if (inlier.size() >= refinedMatches.size() && inlierError <= refinedError) {
//                size_t prevNumInliers = refinedMatches.size();
//                assert(inlierError >= 0);
//                refinedTransformation = transformation;
//                refinedMatches = inlier;
//                refinedError = inlierError;

//                if (inlier.size() == prevNumInliers)
//                    break;
//            } else
//                break;
//        }

//        if (refinedMatches.size() > 0) {
//            validIts++;

//            if (refinedError <= rmse && refinedMatches.size() >= matches.size() && refinedMatches.size() >= minInlierThreshold) {
//                rmse = refinedError;
//                resultingTransformation = refinedTransformation;
//                matches.assign(refinedMatches.begin(), refinedMatches.end());

//                if (refinedMatches.size() > initialMatches12.size() * 0.5)
//                    n += 10;
//                if (refinedMatches.size() > initialMatches12.size() * 0.75)
//                    n += 10;
//                if (refinedMatches.size() > initialMatches12.size() * 0.8)
//                    break;
//            }
//        }
//    }

//    if (validIts == 0) {
//        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
//        vector<cv::DMatch> inlier;
//        ComputeInliersAndError(pF1, pF2, initialMatches12, transformation, minInlierThreshold,
//            inlier, inlierError, maxDistM * maxDistM);

//        if (inlier.size() > minInlierThreshold && inlierError < maxDistM) {
//            assert(inlierError >= 0);
//            resultingTransformation = transformation;
//            matches.assign(inlier.begin(), inlier.end());
//            rmse += inlierError;
//            validIts++;
//        }
//    }

//    bool enoughAbsolute = matches.size() >= minInlierThreshold;
//    return enoughAbsolute;
//}

//vector<cv::DMatch> SampleMatches(uint sampleSize, vector<cv::DMatch>& matches)
//{
//    set<vector<cv::DMatch>::size_type> sampledIds;
//    int safetyNet = 0;

//    while (sampledIds.size() < sampleSize && matches.size() >= sampleSize) {
//        int id1 = rand() % matches.size();
//        int id2 = rand() % matches.size();

//        if (id1 > id2)
//            id1 = id2;

//        sampledIds.insert(id1);

//        if (++safetyNet > 10000)
//            break;
//    }

//    vector<cv::DMatch> sampledMatches;
//    sampledMatches.reserve(sampledIds.size());
//    for (const auto& id : sampledIds)
//        sampledMatches.push_back(matches[id]);

//    return sampledMatches;
//}

//Eigen::Matrix4f GetTransformFromMatches(Frame* pF1, Frame* pF2, vector<cv::DMatch>& m12, bool& valid, const float maxDistM)
//{
//    pcl::TransformationFromCorrespondences tfc;
//    valid = true;
//    float weight = 1.0;

//    for (const auto& m : m12) {
//        cv::Point3f from = pF1->kps3Dc[m.queryIdx];
//        cv::Point3f to = pF2->kps3Dc[m.trainIdx];

//        if (isnan(from.z) || isnan(to.z))
//            continue;

//        weight = 1.0 / (from.z * to.z);
//        tfc.add(Eigen::Vector3f(from.x, from.y, from.z), Eigen::Vector3f(to.x, to.y, to.z), weight);
//    }

//    return tfc.getTransformation().matrix();
//}

//void ComputeInliersAndError(Frame* pF1, Frame* pF2, vector<cv::DMatch>& m12, Eigen::Matrix4f& transformation4f, size_t minInliers, vector<cv::DMatch>& inliers, double& retMeanError, double squaredMaxInlierDistM)
//{
//    inliers.clear();
//    assert(m12.size() > 0);
//    inliers.reserve(m12.size());
//    double meanError = 0.0;
//    Eigen::Matrix4d transformation4d = transformation4f.cast<double>();

//    for (int i = 0; i < m12.size(); ++i) {
//        const cv::DMatch& m = m12[i];
//        const cv::Point3f& origin = pF1->kps3Dc[m12[i].queryIdx];
//        const cv::Point3f& target = pF2->kps3Dc[m12[i].trainIdx];

//        if (origin.z == 0.0 || target.x == 0.0)
//            continue;

//        double mahalDist = ErrorFunction2(Eigen::Vector4f(origin.x, origin.y, origin.z, 1.0), Eigen::Vector4f(target.x, target.y, target.z, 1.0), transformation4d);
//        if (mahalDist > squaredMaxInlierDistM)
//            continue;
//        if (!(mahalDist >= 0.0))
//            continue;

//        meanError += mahalDist;
//        inliers.push_back(m);
//    }

//    if (inliers.size() < 3)
//        retMeanError = 1e9;
//    else {
//        meanError /= inliers.size();
//        retMeanError = sqrt(meanError);
//    }
//}

//double ErrorFunction2(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2, const Eigen::Matrix4d& transformation)
//{
//    static const double cam_angle_x = 58.0 / 180.0 * M_PI;
//    static const double cam_angle_y = 45.0 / 180.0 * M_PI;
//    static const double cam_resol_x = 640;
//    static const double cam_resol_y = 480;
//    static const double raster_stddev_x = 3 * tan(cam_angle_x / cam_resol_x);
//    static const double raster_stddev_y = 3 * tan(cam_angle_y / cam_resol_y);
//    static const double raster_cov_x = raster_stddev_x * raster_stddev_x;
//    static const double raster_cov_y = raster_stddev_y * raster_stddev_y;
//    static const bool use_error_shortcut = true;

//    bool nan1 = std::isnan(x1(2));
//    bool nan2 = std::isnan(x2(2));
//    if (nan1 || nan2)
//        return std::numeric_limits<double>::max();

//    Eigen::Vector4d x_1 = x1.cast<double>();
//    Eigen::Vector4d x_2 = x2.cast<double>();

//    Eigen::Matrix4d tf_12 = transformation;
//    Eigen::Vector3d mu_1 = x_1.head<3>();
//    Eigen::Vector3d mu_2 = x_2.head<3>();
//    Eigen::Vector3d mu_1_in_frame_2 = (tf_12 * x_1).head<3>(); // μ₁⁽²⁾  = T₁₂ μ₁⁽¹⁾

//    if (use_error_shortcut) {
//        double delta_sq_norm = (mu_1_in_frame_2 - mu_2).squaredNorm();
//        double sigma_max_1 = std::max(raster_cov_x, DepthCovariance(mu_1(2)));
//        double sigma_max_2 = std::max(raster_cov_x, DepthCovariance(mu_2(2)));

//        if (delta_sq_norm > 2.0 * (sigma_max_1 + sigma_max_2))
//            return std::numeric_limits<double>::max();
//    }

//    Eigen::Matrix3d rotation_mat = tf_12.block(0, 0, 3, 3);

//    //Point 1
//    Eigen::Matrix3d cov1 = Eigen::Matrix3d::Zero();
//    cov1(0, 0) = raster_cov_x * mu_1(2);
//    cov1(1, 1) = raster_cov_y * mu_1(2);
//    cov1(2, 2) = DepthCovariance(mu_1(2));

//    //Point2
//    Eigen::Matrix3d cov2 = Eigen::Matrix3d::Zero();
//    cov2(0, 0) = raster_cov_x * mu_2(2);
//    cov2(1, 1) = raster_cov_y * mu_2(2);
//    cov2(2, 2) = DepthCovariance(mu_2(2));

//    Eigen::Matrix3d cov1_in_frame_2 = rotation_mat.transpose() * cov1 * rotation_mat;

//    // Δμ⁽²⁾ =  μ₁⁽²⁾ - μ₂⁽²⁾
//    Eigen::Vector3d delta_mu_in_frame_2 = mu_1_in_frame_2 - mu_2;
//    if (std::isnan(delta_mu_in_frame_2(2)))
//        return std::numeric_limits<double>::max();

//    // Σc = (Σ₁ + Σ₂)
//    Eigen::Matrix3d cov_mat_sum_in_frame_2 = cov1_in_frame_2 + cov2;
//    //ΔμT Σc⁻¹Δμ
//    double sqrd_mahalanobis_distance = delta_mu_in_frame_2.transpose() * cov_mat_sum_in_frame_2.llt().solve(delta_mu_in_frame_2);

//    if (!(sqrd_mahalanobis_distance >= 0.0))
//        return std::numeric_limits<double>::max();

//    return sqrd_mahalanobis_distance;
//}

//double DepthCovariance(double depth)
//{
//    static double stddev = DepthStdDev(depth);
//    static double cov = stddev * stddev;
//    return cov;
//}

//double DepthStdDev(double depth)
//{
//    // From Khoselham and Elberink?
//    // Factor c for the standard deviation of depth measurements: sigma_Z = c * depth * depth. Khoshelham 2012 (0.001425) seems to be a bit overconfident."
//    static double depth_std_dev = 0.01;

//    // Previously used 0.006 from information on http://www.ros.org/wiki/openni_kinect/kinect_accuracy;
//    // ...using 2sigma = 95%ile
//    //static const double depth_std_dev  = 0.006;
//    return depth_std_dev * depth * depth;
//}

//////////////////////////////////////////

//            else {

//                pcl::CorrespondencesPtr inliers(new pcl::Correspondences);

//                pcl::PointCloud<pcl::PointXYZ>::Ptr prevCloud(new pcl::PointCloud<pcl::PointXYZ>);
//                pcl::PointCloud<pcl::PointXYZ>::Ptr currCloud(new pcl::PointCloud<pcl::PointXYZ>);
//                pcl::PointCloud<pcl::PointXYZ>::Ptr prevAligned(new pcl::PointCloud<pcl::PointXYZ>);

//                Ransac3D(prevFrame, currFrame, m12, inliers, prevCloud, currCloud);
//                Eigen::Matrix4f initTEigen = Converter::toMatrix4f(initT);
//                pcl::transformPointCloud(*prevCloud, *prevAligned, initTEigen);

//                float d = 0.0f;
//                for (size_t i = 0; i < currCloud->size(); ++i) {
//                    //                    const pcl::Correspondence corr = inliers->at(i);
//                    pcl::PointXYZ transP = prevAligned->points[/*corr.index_query*/ i];
//                    pcl::PointXYZ tgtP = currCloud->points[/*corr.index_match*/ i];

//                    // Euclidean distance
//                    d += (Eigen::Vector3f(transP.x, transP.y, transP.z) - Eigen::Vector3f(tgtP.x, tgtP.y, tgtP.z)).norm();
//                }

//                if (d / inliers->size() > 0.03f) {
//                    //                    Eigen::Matrix4f svdT = Eigen::Matrix4f::Identity();
//                    //                    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>::Ptr
//                    //                    svd(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>);
//                    //                    svd->estimateRigidTransformation(*prevAligned, *currCloud, *inliers, svdT);

//                    //                    initT = initT * Converter::toCvMat(svdT);

//                    if (prevAligned->size() >= 20) {
//                        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
//                        icp.setMaxCorrespondenceDistance(0.005);
//                        icp.setTransformationEpsilon(1e-9);
//                        icp.setMaximumIterations(10);

//                        icp.setInputSource(prevAligned);
//                        icp.setInputTarget(currCloud);

//                        pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
//                        icp.align(*out);

//                        if (icp.hasConverged())
//                            initT = initT * Converter::toCvMat(icp.getFinalTransformation());
//                    }
//                }
//            }

// curr -> prev
// initT = initT.inv();

//            Eigen::Matrix4f trans = Converter::toMatrix4f(initT);
//            PointCloud::Ptr out(new PointCloud);
//            pcl::transformPointCloud(*prevFrame->cloud, *out, trans);

//            viewer.removePointCloud("tgt");
//            viewer.removePointCloud("trans");
//            viewer.addPointCloud(currFrame->cloud, "tgt");
//            viewer.addPointCloud(out, "trans");
//            viewer.spinOnce();

/*
// Generate the matches pointcloud
pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr tgtCloud(new pcl::PointCloud<pcl::PointXYZ>);

vector<cv::Point2i> corrQ, corrT;
cv::Point2i pQ, pT;
for (int i = 0; i < m12.size(); ++i) {
    pQ.x = int(currFrame->kps[m12[i].queryIdx].pt.x + 0.5);
    pQ.y = int(currFrame->kps[m12[i].queryIdx].pt.y + 0.5);

    pT.x = int(prevFrame->kps[m12[i].trainIdx].pt.x + 0.5);
    pT.y = int(prevFrame->kps[m12[i].trainIdx].pt.y + 0.5);

    corrQ.push_back(pQ);
    corrT.push_back(pT);
}

pcl::PointXYZ srcP, tgtP;
for (int i = 0; i < corrQ.size(); ++i) {
    float zQ = currFrame->depth.at<float>(corrQ[i].y, corrQ[i].x);
    float zT = prevFrame->depth.at<float>(corrT[i].y, corrT[i].x);

    if (zQ <= 0 || zT <= 0)
        continue;

    srcP.z = zQ;
    srcP.x = (corrQ[i].x - cx) * zQ * invfx;
    srcP.y = (corrQ[i].y - cy) * zQ * invfy;
    srcCloud->points.push_back(srcP);

    tgtP.z = zT;
    tgtP.x = (corrT[i].x - cx) * zT * invfx;
    tgtP.y = (corrT[i].y - cy) * zT * invfy;
    tgtCloud->points.push_back(tgtP);
}

srcCloud->height = 1;
srcCloud->width = srcCloud->points.size();
srcCloud->is_dense = false;

tgtCloud->height = 1;
tgtCloud->width = tgtCloud->points.size();
tgtCloud->is_dense = false;

// Ransac use keypoint cloud
pcl::CorrespondencesPtr correspondenceIn(new pcl::Correspondences);
pcl::CorrespondencesPtr correspondenceOut(new pcl::Correspondences);

for (int i = 0; i < srcCloud->size(); ++i) {
    pcl::Correspondence corrIn;
    corrIn.index_query = i;
    corrIn.index_match = i;
    corrIn.weight = 1.0 / (srcCloud->points[i].z * tgtCloud->points[i].z);
    correspondenceIn->push_back(corrIn);
}

pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>::Ptr
    ransac(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>);
ransac->setInputSource(srcCloud);
ransac->setInputTarget(tgtCloud);
ransac->setInlierThreshold(0.05);
ransac->setMaximumIterations(1000);
ransac->getRemainingCorrespondences(*correspondenceIn, *correspondenceOut);
Eigen::Matrix4f initialT = ransac->getBestTransformation();

pcl::PointCloud<pcl::PointXYZ>::Ptr alignedSrc(new pcl::PointCloud<pcl::PointXYZ>);
pcl::transformPointCloud(*srcCloud, *alignedSrc, initialT);

// Compute RT matrix
//            Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
//            pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>::Ptr
//            svd(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>);
//            svd->estimateRigidTransformation(*srcCloud, *tgtCloud, *correspondenceOut, Rt);
pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
icp.setMaxCorrespondenceDistance(0.05);
icp.setRANSACOutlierRejectionThreshold(0.05);
icp.setRANSACIterations(100);
icp.setMaximumIterations(20);

icp.setInputSource(alignedSrc);
icp.setInputTarget(tgtCloud);

pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
icp.align(*out);

Eigen::Matrix4f refinedT = icp.getFinalTransformation() * initialT;

//            p->removePointCloud("source");
//            p->removePointCloud("target");
//            p->addPointCloud(out, "target", vp2);
//            p->addPointCloud(prevFrame->cloud, "source", vp2);
//            p->spinOnce(1);

Tcw = prevFrame->mTcw * Converter::toCvMat(refinedT);
*/
