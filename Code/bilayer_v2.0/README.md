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

Inputs: `../Inputs/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy` and `../Inputs/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy`
All datasets are with (7.7,106) moiré potential at K
Datasets at theta_deviation=0:
    - Large grid   (127 chunks): 1:25/25,  160:180/21,  -2:2/41,            -2:2/41
        +   Symmetry of wp(d)->-wp(d)
        +   2 minima
    - Medium grid  (127 chunks): 5:22/18,  160:180/21,  -2.5:-1.92/30,      0.3:1.3/51
        +   Made a mistake in the wp range, should have been -1.29
        +   Only one of the 2 peaks is found, the lower one: (-2.1,0.53)
    - Small grid 1 (127 chunks): 10:22/13, 165:180/16,  -2:-2.2/41,         0.4:0.6/41
        +   Minimum at (-2.075,0.525)
    - Small grid 2 (127 chunks): 10:22/13, 165:180/16,  -1.4:-1.6/41,       1.05:1.25/41
        +   Minimum at (-1.555,1.145)
    - Fine grid 1 (127 chunks):  16:21/51, 165:180/16,  -2.090:-2.065/26,   0.520:0.535/16
        +   Minimum at (wp,wd)=(-2.075 eV, 0.525 eV) and (V,phi)=(19.6 meV, 173°)
    - Fine grid 2 (127 chunks):  15:20/51, 165:180/16,  -1.565:-1.540/26,   1.135:1.150/16
        +   Minimum at (wp,wd)=(-1.556 eV, 1.143 eV) and (V,phi)=(16.5 meV, 175°)
    - All phases  (300 chunks):  15:20/51, 1:359/360,   -1.565:-1.550/16,   1.135:1.150/16
        +   Periodicity of 120° as expected
        +   Should have considered a larger lower bound on V -> ~10 probably enough
    - Paper plot  (300 chunks):  10:20/51, 1:359/360    -1.580:-1.530/11,   1.120:1.170/11
        +   

### K
In this situation there is no role of interlayer coupling, so we just look at the distance between main band and moiré bands.
From the ARPES exp we have, for S11 [eV]:
    - d=0.170 eV
    - p=-0.8990 eV      (units of S11)
    - p=-0.4290 eV      (units of S3, offset of 0.47 eV)

Inputs: `../Inputs/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy` and `../Inputs/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy`
All datasets are with (16.5,175) moiré at G and (-1.556,1.143) intelayer coupling
Datasets:
    - Theta_dev=0, 0.001:0.020/20,   0:359/360
        +   120° periodicity centered around 120-240 and 360
        +    Minimum increasing with V -> in contraddiction with gap at some point
    - Theta_dev=0,    0.001:0.020/191,   0:359/360
        +   120° periodicity centered around 120-240 and 360
        +   More precise version of the above, same results
    - Theta_dev=0.3,  0.001:0.020/191,   0:359/360
        +   120° periodicity centered around 120-240 and 360
        +   No good results, measure is quite distant from ARPES
    - Theta_dev=-0.3, 0.001:0.020/191,   0:359/360
        +   120° periodicity centered around 120-240 and 360
        +   Parabolic minimum wrt ARPES

## Gap
We compute also the gap values, in particular to constrain the moirè potential and phase at K.
`gap.py`


### TODO
- Nice EDC plot for paper

