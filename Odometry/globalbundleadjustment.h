#ifndef GLOBALBUNDLEADJUSTMENT_H
#define GLOBALBUNDLEADJUSTMENT_H

#include <vector>

class Map;
class KeyFrame;
class Landmark;

class GlobalBundleAdjustment {
public:
    static void Compute(Map* pMap, int nIterations = 5, bool* pbStopFlag = NULL, const unsigned long nLoopKF = 0,
        const bool bRobust = true);

    static void BundleAdjustment(const std::vector<KeyFrame*>& vpKF, const std::vector<Landmark*>& vpLM, int nIterations = 5,
        bool* pbStopFlag = NULL, const unsigned long nLoopKF = 0, const bool bRobust = true);
};

#endif // GLOBALBUNDLEADJUSTMENT_H
