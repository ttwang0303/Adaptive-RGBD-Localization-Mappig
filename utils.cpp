#include "utils.h"
#include "converter.h"
#include "frame.h"
#include <fstream>
#include <sstream>

using namespace std;

void LoadImages(const string& associationFilename, vector<string>& vImageFilenamesRGB,
    vector<string>& vImageFilenamesD, vector<double>& vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(associationFilename.c_str());
    while (!fAssociation.eof()) {
        string s;
        getline(fAssociation, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vImageFilenamesD.push_back(sD);
        }
    }
}

vector<cv::DMatch> Match(Frame* pF1, Frame* pF2, cv::Ptr<cv::DescriptorMatcher> pMatcher)
{
    vector<vector<cv::DMatch>> matchesKnn;
    vector<cv::DMatch> m12;

    pMatcher->knnMatch(pF1->desc, pF2->desc, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < 0.8 * m2.distance) {
            m12.push_back(m1);
        }
    }

    return m12;
}

void DrawMatches(Frame* pF1, Frame* pF2, const vector<cv::DMatch>& m12, const int delay)
{
    cv::Mat out;
    cv::drawMatches(pF1->im, pF1->kps, pF2->im, pF2->kps, m12, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches", out);
    cv::waitKey(delay);
}
