# What's what

Here we compute CEM around K. We do it in two different ways: with and without interlayer.
Why?
Because in case of 0° rotation, the two layers have K on top of K (referred to as parallel 'P') but in case of 60° rotation K is on 
top of K' (anti-parallel 'AP'). In both cases we expect C3 symmetry because of the BZ shape (K to K/K' or to Gamma). For P there is 
in principle more interlayer coupling since the spin is locked to the valley and the two valleys have the same spin. Instead for AP
the interlayer should be reduced.

In this code we use the parabolic single orbital ('simple') model, with and without interlayer coupling.

We use a Hamiltonian of the form:
H = (-k^2/2m1       t(k)            )       with t(k) to be defined and a diagonal energy shift mu.
    (t*(k)          -k^2/2m2 - c    )

In the P case we fit the image using a. In the AP case instead we put a=b=0.

S11 seems to have C6 symmetry, with intensities probably modulated by the matrix elements. This could be due to weak interlayer coupling -> AP case.
S3 seems to have a more clear C3 symmetry. This could be due to more relevant interlayer coupling -> P case.

STEPS:
1-Extract band parameters m1 and m2 for WSe2 and WS2, respectively. We take the monolayers' data around K to do so. Just the tip of the band is enough, since the 
    cuts are 90 to 160 meV from the top valence band (TVB), which is given by WSe2. 
    DONE -> Used a range of 0.25 from VBM. 
            Used data from both G and M -> double fit with different m for the 2 sides.
2-Compute CEM for cuts every 5 meV from TVB without interlayer coupling (IC).
3-Add IC in the form of:    
        a constant: t(k) = -a -> compute some cases (see value of a around Gamma for a starting value);
        a constant plus a squared k dependence which is 0 at k: t(k) = -a(1-b*k^2) -> compute some cases (again look at the value of a and b from Gamma for reference);

