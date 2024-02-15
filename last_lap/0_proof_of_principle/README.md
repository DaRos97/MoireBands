# Purpose
Here we take the interlayer interaction at the Moire level to see if we can get the different symmetries around Gamma
of the side bands.

We have parallel (P) and anti-parallel (AP) stacking. We expect respectively 6-fold and 3-fold symmetry, so images of 
samples 11 and 3.

We use a 2 band model, one for layer, with standard quadratic energy dependence (-k^2/2m).

The interlayer is in two parts: on BZ level and on Moire level.

The first is on the big-BZ level, and can be:
    - P:  t0(k) = a+b sum {j=1 to 6} e {ik a}   (a is exagonal lattice vector)
    - AP: t0(k) = b sum {j=1 to 3} e {ik d} (d is vector connecting 1st nn)

The only way to get 3-fold symmetry seems to be through moire physics, i.e. add spacial dependence to the interlayer
interaction. 

The Moire potential connects mini-BZs at neighboring reciprocal moire vectors. Together with this we add an interlayer
coupling which will be different for P and AP case:
    - P:  t1(r) = f sum {j=1 to 6} e {iGM r} -> this means that there will be f connecting mini-BZ connected by GM
    - AP: t1(r) = f1 sum {j=1,3,5} e {iGM r} + f2 sum {j=2,4,6} e {iGM r} -> same as above but different strength 
        for different reciprocal moire vectors.

In first approximation we take the moire interlayer to be constant in k.

Overall we have parameters: 
    - P:  m1,m2, a,b,c, f
    - AP: m1,m2, b,c,   f1,f2
m is the mass of the bands. See previous calculation to see the values similar to S11 and S3 close to Gamma.
c is an offset in one of the two bands.

# Code
We need to do all the steps. Construct big Hamiltonian with moire copies and project on first BZ.
Apply minimal spreading to see the differences as clear as possible.
Tune mostly f1,f2 wrt f.

