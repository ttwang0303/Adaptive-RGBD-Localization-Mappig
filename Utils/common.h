#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <iostream>
#include <memory>
#include <string>

template <typename T>
using Ptr = std::shared_ptr<T>;

#define endll endl \
    << endl // double end line definition

const std::string RESET = "\033[0m";
const std::string BLACK = "\033[30m";
const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN = "\033[36m";
const std::string WHITE = "\033[37m";
const std::string BOLDBLACK = "\033[1m\033[30m";
const std::string BOLDRED = "\033[1m\033[31m";
const std::string BOLDGREEN = "\033[1m\033[32m";
const std::string BOLDYELLOW = "\033[1m\033[33m";
const std::string BOLDBLUE = "\033[1m\033[34m";
const std::string BOLDMAGENTA = "\033[1m\033[35m";
const std::string BOLDCYAN = "\033[1m\033[36m";
const std::string BOLDWHITE = "\033[1m\033[37m";

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

// -------- D1 --------
//const double fx = 468.6;
//const double fy = 468.61;
//const double cx = 318.27;
//const double cy = 243.99;

// Common
const double depthFactor = 1.0 / 5000.0;
const double invfx = 1.0 / fx;
const double invfy = 1.0 / fy;

const int nFeatures = 1000;

// Viewer
const bool DRAW_LANDMARKS = true;
const bool DRAW_KFS = true;
const bool DRAW_DENSECLOUD = false;

// Iaicp
//const int height = 480;
//const int width = 640;

#endif // CONSTANTS_H
