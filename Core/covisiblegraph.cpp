#include "covisiblegraph.h"
#include "keyframe.h"
#include "landmark.h"

using namespace std;

CovisibilityGraph::CovisibilityGraph()
    : mbFirstConnection(true)
    , mpParent(nullptr)
{
}

CovisibilityGraph::CovisibilityGraph(KeyFrame* pKFroot)
    : mpKFroot(pKFroot)
    , mbFirstConnection(true)
    , mpParent(nullptr)
{
}

void CovisibilityGraph::SetRootNode(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    mpKFroot = pKF;
}

void CovisibilityGraph::AddNode(KeyFrame* pKFnode, const int& weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (!mTree.count(pKFnode))
            mTree[pKFnode] = weight;
        else if (mTree[pKFnode] != weight)
            mTree[pKFnode] = weight;
        else
            return;
    }

    UpdateBestNodes();
}

void CovisibilityGraph::UpdateBestNodes()
{
    unique_lock<mutex> lock(mMutexConnections);

    vector<pair<int, KeyFrame*>> vOrderedByWeight;
    vOrderedByWeight.reserve(mTree.size());

    for (auto it = mTree.begin(); it != mTree.end(); it++) {
        KeyFrame* pKF = it->first;
        int w = it->second;
        vOrderedByWeight.push_back({ w, pKF });
    }

    // Sort by weight
    std::sort(vOrderedByWeight.begin(), vOrderedByWeight.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for (vector<pair<int, KeyFrame*>>::const_iterator it = vOrderedByWeight.begin(); it != vOrderedByWeight.end(); it++) {
        const int& w = it->first;
        KeyFrame* pKF = it->second;
        lKFs.push_front(pKF);
        lWs.push_front(w);
    }

    mvpOrderedKFs = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

void CovisibilityGraph::UpdateConnections()
{
    map<KeyFrame*, int> KFcounter;
    vector<Landmark*> vpLM = mpKFroot->GetLandmarksMatched();
    long unsigned int KFid = mpKFroot->GetId();

    // For all map points in root keyframe check in which other keyframes are they seen
    // Increase counter for those keyframes
    for (Landmark* pLM : vpLM) {
        if (!pLM)
            continue;
        if (pLM->isBad())
            continue;
        pLM->Covisibility(KFcounter, KFid);
    }

    if (KFcounter.empty())
        return;

    // If the counter is greater than threshold add connection
    // In case no keyframe counter is over threshold add the one with maximum counter
    int nmax = 0;
    KeyFrame* pKFmax = nullptr;
    int th = 15;

    vector<pair<int, KeyFrame*>> vPairs;
    vPairs.reserve(KFcounter.size());
    for (auto& [pKF, n] : KFcounter) {
        if (n > nmax) {
            nmax = n;
            pKFmax = pKF;
        }
        if (n >= th) {
            vPairs.push_back({ n, pKF });
            pKF->mG.AddNode(mpKFroot, n);
        }
    }

    if (vPairs.empty()) {
        vPairs.push_back({ nmax, pKFmax });
        pKFmax->mG.AddNode(mpKFroot, nmax);
    }

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for (auto& [w, pKF] : vPairs) {
        lKFs.push_front(pKF);
        lWs.push_front(w);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mTree = KFcounter;
        mvpOrderedKFs = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if (mbFirstConnection && KFid != 0) {
            mpParent = mvpOrderedKFs.front();
            mpParent->mG.AddChild(mpKFroot);
            mbFirstConnection = false;
        }
    }
}

set<KeyFrame*> CovisibilityGraph::GetNodeSet()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> sNodes;
    for (auto it = mTree.begin(); it != mTree.end(); it++) {
        KeyFrame* pKF = it->first;
        sNodes.insert(pKF);
    }

    return sNodes;
}

vector<KeyFrame*> CovisibilityGraph::GetOrderedNodes()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedKFs;
}

void CovisibilityGraph::EraseNode(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mTree.count(pKF)) {
            mTree.erase(pKF);
            bUpdate = true;
        }
    }

    if (bUpdate)
        UpdateBestNodes();
}

vector<KeyFrame*> CovisibilityGraph::GetBestNodes(const int& N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if ((int)mvpOrderedKFs.size() < N)
        return mvpOrderedKFs;
    else
        return vector<KeyFrame*>(mvpOrderedKFs.begin(), mvpOrderedKFs.begin() + N);
}

vector<KeyFrame*> CovisibilityGraph::GetNodesByWeight(const int& w)
{
    unique_lock<mutex> lock(mMutexConnections);
    if (mvpOrderedKFs.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = std::upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, weightComp);
    if (it == mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else {
        const auto n = it - mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedKFs.begin(), mvpOrderedKFs.begin() + n);
    }
}

int CovisibilityGraph::GetWeight(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if (mTree.count(pKF))
        return mTree[pKF];
    else
        return 0;
}

void CovisibilityGraph::AddChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void CovisibilityGraph::EraseChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void CovisibilityGraph::ChangeParent(KeyFrame* pKF)
{
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mpParent = pKF;
    }
    pKF->mG.AddChild(mpKFroot);
}

set<KeyFrame*> CovisibilityGraph::GetChildSet()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* CovisibilityGraph::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool CovisibilityGraph::hasChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void CovisibilityGraph::AddLoopEdge(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpKFroot->SetNotErase();
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> CovisibilityGraph::GetLoopSet()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

bool CovisibilityGraph::isLoopEmpty()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges.empty();
}
