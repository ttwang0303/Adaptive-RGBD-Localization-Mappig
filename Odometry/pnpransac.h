#ifndef PNPRANSAC_H
#define PNPRANSAC_H

class Frame;

class PnPRansac {
public:
    static int Compute(Frame& frame);
};

#endif // PNPRANSAC_H
