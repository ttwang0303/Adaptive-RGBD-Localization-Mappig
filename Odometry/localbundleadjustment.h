#ifndef LOCALBUNDLEADJUSTMENT_H
#define LOCALBUNDLEADJUSTMENT_H

class KeyFrame;
class Map;

class LocalBundleAdjustment {
public:
    static void Compute(KeyFrame* pKF, bool* pbStopFlag, Map* pMap);
};

#endif // LOCALBUNDLEADJUSTMENT_H
