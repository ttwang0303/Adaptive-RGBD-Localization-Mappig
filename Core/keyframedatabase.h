#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include "DBoW3/DBoW3.h"
#include "DBoW3/QueryResults.h"
#include <mutex>

class KeyFrame;

class Database {
public:
    Database(DBoW3::Vocabulary* pVoc);

    void Add(KeyFrame* pKF);

    void Erase(KeyFrame* pKF);

    void Clear();

    std::vector<KeyFrame*> Query(KeyFrame* pKF, float minScore);

protected:
    DBoW3::Vocabulary* mpVoc;
    std::vector<std::list<KeyFrame*>> mvInvertedFile;
    std::mutex mMutex;
};

#endif // KEYFRAMEDATABASE_H
