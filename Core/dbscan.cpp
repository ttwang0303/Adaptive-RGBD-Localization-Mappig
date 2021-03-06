#include "dbscan.h"
#include <iostream>

using namespace std;

DBScan::DBScan(double _eps, int _minPts, int _featuresFromCluster)
{
    eps = _eps;
    minPts = _minPts;
    featuresFromCluster = _featuresFromCluster;
}

void DBScan::expandCluster(vector<int> neighbourList, int clusteringSetSize, int& C)
{
    // testing the neighbours
    for (vector<int>::size_type j = 0; j < neighbourList.size(); j++) {
        int x = neighbourList[j];

        // If not visited
        if (visited[x] != true) {
            visited[x] = true;
            vector<int> neighbourNeighbourList;

            // Calculating the number of neighbours
            for (int k = 0; k < clusteringSetSize; k++) {
                if (!visited[k] && dist[min(x, k)][max(x, k)] < eps) {
                    neighbourNeighbourList.push_back(k);
                }
            }

            // If it has enough neighbours it's neighbours can be checked
            // Merging ...
            if ((int)neighbourNeighbourList.size() >= minPts) {
                neighbourList.insert(neighbourList.end(), neighbourNeighbourList.begin(), neighbourNeighbourList.end());
            }
        }

        // if it is not yet labeled
        if (cluster[x] == 0)
            cluster[x] = C;
    }
}

int DBScan::findingClusters(int clusteringSetSize)
{
    // Starting cluster id
    int C = 1;

    // For all points
    for (int i = 0; i < clusteringSetSize; i++) {
        if (visited[i] != true) {
            visited[i] = true;
            vector<int> neighbourList;
            // Finding neighbours
            for (int k = 0; k < clusteringSetSize; k++) {
                if (dist[min(i, k)][max(i, k)] < eps) {
                    neighbourList.push_back(k);
                }
            }
            // If there are not enough neighbours to form a cluster
            if ((int)neighbourList.size() < minPts)
                cluster[i] = -1;
            else {
                // There is a need cluster!
                cluster[i] = C;
                expandCluster(neighbourList, clusteringSetSize, C);
                C++;
            }
        }
    }
    return C;
}

void DBScan::run(vector<cv::KeyPoint>& clusteringSet)
{
    int clusteringSetSize = (int)clusteringSet.size();

    // Calculating similarity matrix
    dist = vector<vector<float>>(clusteringSetSize,
        vector<float>(clusteringSetSize, 0));

    for (int i = 0; i < clusteringSetSize; i++)
        for (int j = i; j < clusteringSetSize; j++)
            dist[i][j] = (float)cv::norm(clusteringSet[i].pt - clusteringSet[j].pt);

    // Preparation - visited nodes information
    visited = vector<bool>(clusteringSetSize, false);

    // Output information
    // -1 means noise
    // 0 not yet processed
    // >0 belongs to group of given id
    cluster = vector<int>(clusteringSetSize);

    // For all points
    int clusterCount = findingClusters(clusteringSetSize);

    // Just leave strongest from each cluster
    vector<int> clusterChosenCount(clusterCount);
    for (int i = 0; i < clusteringSetSize; i++) {
        int clusterId = cluster[i];
        if (clusterId > 0) {
            if (clusterChosenCount[clusterId] > featuresFromCluster - 1) {
                if (clusteringSet[i].octave == -5)
                    cout << "DBScan issue : SHIT :/ Octave == -5 -> need to find another way of marking to erase" << endl;
                clusteringSet[i].octave = -5;
            } else
                clusterChosenCount[clusterId]++;
        }
    }

    // Remove those bad elements
    clusteringSet.erase(remove_if(clusteringSet.begin(), clusteringSet.end(), [](cv::KeyPoint& kp) { return kp.octave == -5; }), clusteringSet.end());
}
