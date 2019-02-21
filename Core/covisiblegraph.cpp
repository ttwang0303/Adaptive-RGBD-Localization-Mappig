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
    unique_lock<mutex> lock(mMutexGraph);
    mpKFroot = pKF;
}

void CovisibilityGraph::AddNode(KeyFrame* pKFnode, const int& weight)
{
    {
        unique_lock<mutex> lock(mMutexGraph);
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
    unique_lock<mutex> lock(mMutexGraph);

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
    vector<Landmark*> vpLM = mpKFroot->GetLandmarks();
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
        unique_lock<mutex> lockCon(mMutexGraph);
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
    unique_lock<mutex> lock(mMutexGraph);
    set<KeyFrame*> sNodes;
    for (auto it = mTree.begin(); it != mTree.end(); it++) {
        KeyFrame* pKF = it->first;
        sNodes.insert(pKF);
    }

    return sNodes;
}

vector<KeyFrame*> CovisibilityGraph::GetOrderedNodes()
{
    unique_lock<mutex> lock(mMutexGraph);
    return mvpOrderedKFs;
}

void CovisibilityGraph::EraseNode(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexGraph);
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
    unique_lock<mutex> lock(mMutexGraph);
    if ((int)mvpOrderedKFs.size() < N)
        return mvpOrderedKFs;
    else
        return vector<KeyFrame*>(mvpOrderedKFs.begin(), mvpOrderedKFs.begin() + N);
}

vector<KeyFrame*> CovisibilityGraph::GetNodesByWeight(const int& w)
{
    unique_lock<mutex> lock(mMutexGraph);
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
    unique_lock<mutex> lock(mMutexGraph);
    if (mTree.count(pKF))
        return mTree[pKF];
    else
        return 0;
}

void CovisibilityGraph::EraseRootConnections()
{
    unique_lock<mutex> lock(mMutexGraph);
    for (auto& [pKF, w] : mTree) {
        pKF->mG.EraseNode(mpKFroot);
    }
}

void CovisibilityGraph::AddChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexGraph);
    mspChildrens.insert(pKF);
}

void CovisibilityGraph::EraseChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexGraph);
    mspChildrens.erase(pKF);
}

void CovisibilityGraph::ChangeParent(KeyFrame* pKF)
{
    {
        unique_lock<mutex> lockCon(mMutexGraph);
        mpParent = pKF;
    }
    pKF->mG.AddChild(mpKFroot);
}

set<KeyFrame*> CovisibilityGraph::GetChildrenSet()
{
    unique_lock<mutex> lockCon(mMutexGraph);
    return mspChildrens;
}

KeyFrame* CovisibilityGraph::GetParent()
{
    unique_lock<mutex> lockCon(mMutexGraph);
    return mpParent;
}

bool CovisibilityGraph::hasChild(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexGraph);
    return mspChildrens.count(pKF);
}

void CovisibilityGraph::UpdateSpanningTree()
{
    unique_lock<mutex> lock(mMutexGraph);
    mTree.clear();
    mvpOrderedKFs.clear();

    set<KeyFrame*> sParentCandidates;
    sParentCandidates.insert(mpParent);

    // Assign at each iteration one children with a parent (the pair with
    // highest covisibility weight) Include that children as new parent
    // candidate for the rest
    while (!mspChildrens.empty()) {
        bool bContinue = false;

        int max = -1;
        KeyFrame* pC;
        KeyFrame* pP;

        for (KeyFrame* pKF : mspChildrens) {
            if (pKF->isBad())
                continue;

            // Check if a parent candidate is connected to the KF
            vector<KeyFrame*> vpConnected = pKF->mG.GetOrderedNodes();
            for (size_t i = 0; i < vpConnected.size(); i++) {
                for (KeyFrame* pKFpc : sParentCandidates) {
                    if (vpConnected[i]->GetId() == pKFpc->GetId()) {
                        int w = pKF->mG.GetWeight(vpConnected[i]);
                        if (w > max) {
                            pC = pKF;
                            pP = vpConnected[i];
                            max = w;
                            bContinue = true;
                        }
                    }
                }
            }
        }

        if (bContinue) {
            pC->mG.ChangeParent(pP);
            sParentCandidates.insert(pC);
            mspChildrens.erase(pC);
        } else
            break;
    }

    // If children has no covisibility links with any parent, assign
    // to the original parent of root KF
    if (!mspChildrens.empty()) {
        for (KeyFrame* pKFc : mspChildrens)
            pKFc->mG.ChangeParent(mpParent);
    }

    mpParent->mG.EraseChild(mpKFroot);
}

void CovisibilityGraph::AddLoopEdge(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexGraph);
    mpKFroot->SetNotErase();
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> CovisibilityGraph::GetLoopSet()
{
    unique_lock<mutex> lockCon(mMutexGraph);
    return mspLoopEdges;
}

bool CovisibilityGraph::isLoopEmpty()
{
    unique_lock<mutex> lockCon(mMutexGraph);
    return mspLoopEdges.empty();
}
