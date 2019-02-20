#ifndef PNPSOLVER_H
#define PNPSOLVER_H

class Frame;

class PnPSolver {
public:
    int static Compute(Frame* pFrame);
};

#endif // PNPSOLVER_H
