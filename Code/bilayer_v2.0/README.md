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

Datasets (all with (7,106) moiré at K):
    - Large grid (128): 1:25/25, 160:180/21, -2:2/41,        -2:2/41    -> minimum close to w1p=-2,w1d=0.5
    - Latge grid (128): 1:25/25, 160:180/21, -3:-1/41,       0.2:0.7/11 -> minimum of position still at the boundary
    - Large grid (128): 5:25/21, 165:180/16, -4:-2.5/76,     0.4:0.6/21 -> still ad boundary of wp
    - Large grid (500): 5:25/11, 166:180/8,  -8:-4/81,       0.4:0.6/21 -> found the wp minimum at -5.88
    - Finer grid (500): 1:25/25, 160:180/21, -6.2:-5.5/71,   0.4:0.6/21 -> best interlayer at (-5.88,0.48), best moiré potential at (17,172:176)
    - Large plot (200): 8:25/18, 0:358/180,  -5.98:-5.78/21, 0.4:0.6/21, theta=2.8 -> too coarse
    - Large plot (250): 1:30/30, 0:359/360,  -5.95:-5.81/15, 0.41:0.55/15, theta=2.8 ->
    - Large plot (250): 1:30/30, 0:359/360,  -5.95:-5.81/15, 0.41:0.55/15, theta=2.5 ->

### K
In this situation there is no role of interlayer coupling, so we just look at the distance between main band and moiré bands.
From the ARPES exp we have, for S11 [eV]:
    - d=0.170 eV
    - p=-0.8990 eV      (units of S11)
    - p=-0.4290 eV      (units of S3, offset of 0.47 eV)

Datasets (all with (17,174) moiré at G and (-5.88,0.48) intelayer):
    - theta = 2.8 (200): 1:30/30, 0:359/360 -> double parabola of minima!
    - theta = 2.5 (80) : 1:30/30, 0:359/360 -> change of phase
    - theta = 3.1 (80) : 1:30/30, 0:359/360 -> parabola shiftedby ~10 meV and no good results

### TODO
- Nice EDC plot for paper
- Solve problem of edc at K

