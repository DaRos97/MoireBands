#Introduction
Here we take the monolayer tight-binding parameters and compute the bilayer bands introducing interlayer coupling and moiré potential.
We perform analysis regarding:
    - EDC
    - LDOS

## EDC
We use the Energy Distance of Crossing bands as an indicator to extract the moiré potential amplitude and phase.
We do this at both Gamma and K.

### Gamma
At Gamma, since it is mostly OPO we also include two interlayer couplings: w1p and w1d.
For the 4-dimensional space of V, phi, w1p and w1d we define a distance from the ARPES result in two ways:
    - Considering distances: d1,d2 distance of side band crossing(sbc) and of WS2 band from TVB, respectively
    - Considering positions: p1,p2,p3 positions of TVB, sbc and WS2, respectively.

From the ARPES exp we have, for S11 [eV]:
    - p1=-1.1599, p2=-1.2531, p3=-1.8200        (units of S11)
    - p1=-0.6899, p2=-0.7831, p3=-1.3500        (units of S3, offset of 0.47 eV)
    - d1=0.0932, d2=0.6601

### K
In this situation there is no role of interlayer coupling, so we just look at the distance between main band and moiré bands.
From the ARPES exp we have, for S11 [eV]:
    - d=0.170 eV
    - p=-0.8990 eV      (units of S11)
    - p=-0.4290 eV      (units of S3, offset of 0.47 eV)

### TODO
- edcG with theta + and - 0.3°:
        - Running now on mafalda
        - Remember to change the angle_deviation in mergeEDC.py
- Nice EDC plot for paper
- Solve problem of edc at K

