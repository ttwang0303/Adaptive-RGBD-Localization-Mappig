#ifndef COVISIBLEGRAPH_H
#define COVISIBLEGRAPH_H

#include <list>
#include <map>
#include <mutex>
#include <set>
#include <vector>

class KeyFrame;

class CovisibilityGraph {
public:
    CovisibilityGraph();
    CovisibilityGraph(KeyFrame* pKFroot);

    void SetRootNode(KeyFrame* pKF);

    // Covisibility graph functions
    void AddNode(KeyFrame* pKF, const int& weight);
    void UpdateBestNodes();
    void UpdateConnections();
    std::set<KeyFrame*> GetNodeSet();
    void EraseNode(KeyFrame* pKF);
    std::vector<KeyFrame*> GetBestNodes(const int& N);
    std::vector<KeyFrame*> GetNodesByWeight(const int& w);
    std::vector<KeyFrame*> GetOrderedNodes();
    int GetWeight(KeyFrame* pKF);
    void EraseRootConnections();

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChildrenSet();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);
    void UpdateSpanningTree();

    // Loop edges functions
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopSet();
    bool isLoopEmpty();

    static bool weightComp(int a, int b) { return a > b; }

protected:
    KeyFrame* mpKFroot;
    std::map<KeyFrame*, int> mTree;
    std::vector<KeyFrame*> mvpOrderedKFs;
    std::vector<int> mvOrderedWeights;

    // Spannig Tree
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;

    // Loop edges
    std::set<KeyFrame*> mspLoopEdges;

    std::mutex mMutexGraph;
};

#endif
