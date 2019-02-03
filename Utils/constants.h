#ifndef CONSTANTS_H
#define CONSTANTS_H

// -------- FR1 --------
const double fx = 517.3;
const double fy = 516.5;
const double cx = 318.6;
const double cy = 255.3;

// -------- FR2 --------
//const double fx = 520.9;
//const double fy = 521.0;
//const double cx = 325.1;
//const double cy = 249.7;

// -------- FR3 --------

// -------- ICL --------
//const double fx = 481.20;
//const double fy = -480.00;
//const double cx = 319.50;
//const double cy = 239.50;

// Common
const double depthFactor = 1.0 / 5000.0;
const double invfx = 1.0 / fx;
const double invfy = 1.0 / fy;

const int nFeatures = 1000;

#endif // CONSTANTS_H
