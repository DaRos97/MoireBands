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
The distance measure

Define measures
Change dirname to include parameters' boundaries and numbers
Change final .h5 file to have all specs in the name
