# Description
Here we compute the final image using the 11-band tight binding Hamiltonian fitted on the monolayer bands
and the interlayer coupling exracted from the bilayer main bands. 
We need to include the Moire periodicity and compute the spreading of the bands in energy and momentum,
so it's a longer calculation.

# Code
get_res_1_2.py  imports the tight binding parameters and the interlayer coupling.
moire.py & functions.py     compute the final image

# Computed
## Bao
    - N=4,px=5, sk=se=0.01, Gauss, DFT:False, phiG:pi, aM:79.8
    - Interlayer: C6-C3, Vg: 0.025-0.03-0.035, Vk: 0.005-0.0077-0.01-0.015-0.02, phiK: 0-(-1.85)-3.14       -> 90 images

# Computing
## Bao
    - N=4, px=5, aM=79.8, interlayer C3. 
    - Vg: 0.015-0.025-0.03-0.035-0.04, phiG: 0:pi:10, Vk: 0.003-0.0077-0.01-0.015-0.02, phiK: 0:pi:10       -> 2500 images

# Plan
Parameters are:
    - TB and interlayer parameters are fixed
    - Type of interlayer: U1-C6-C3
    - Moire parameters: N, Vg, phig, Vk, phik, moire lattice constant
    - pixel factor 
Essentially what I can vary now is just the (V,phi) at gamma and K. 
Compute large number of figures for broad range of moire potentials. 
Keep N=4, px=5, aM=79.8, interlayer C3.

Since it is a large number of figures, maybe good to define a measure of closeness to
the experimental figure. How?

If it does not give good positions of side bands (no not care too much about intensity),
let's consider changing also the interlayer parameters.


# Results
Near K: 
    - (0.01-0) -> good first side band and start of second
    - (0.01,5/9pi) -> also good
    - (0.015,8/9pi&pi) -> ok
    - (0.015,6/9pi) -> better but has gap at k
Near Gamma:
phase pi&3/9pi is the only without sizeable gap at gamma -> probably same image because of some symmetry.
Amplitude either 0.03 or 0.035 but side bands at right position are a bit faint.

K does not influence at all the bands near Gamma.

Main bands are a bit modified by moire potential.

# To check

HSO
