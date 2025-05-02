# What's what

Here we compute CEM around K. We do it in two different ways: with and without interlayer.
Why?
Because in case of 0° rotation, the two layers have K on top of K (referred to as parallel 'P') but in case of 60° rotation K is on 
top of K' (anti-parallel 'AP'). In both cases we expect C3 symmetry because of the BZ shape (K to M or to Gamma). For P there is 
in principle more interlayer coupling since the spin is locked to the valley and the two valleys have the same spin. Instead for AP
the interlayer should be reduced.

In this code we use the parabolic single orbital ('simple') model, with and without interlayer coupling.

For the bilayer we use a Hamiltonian of the form:
H = (E up           t(k)            )       with t(k) = -a(1-b k^2) to be defined and a diagonal energy shift mu. and E up/down defined below.
    (t*(k)          E down - c      )

In the P case we fit the image using a and b. In the AP case instead we put a=b=0.

S11 seems to have C6 symmetry, with intensities probably modulated by the matrix elements. This could be due to weak interlayer coupling -> AP case.
S3 seems to have a more clear C3 symmetry. This could be due to more relevant interlayer coupling -> P case.

STEPS:
Procedure A:
    1-Extract band parameters for WSe2 and WS2, respectively. We take the monolayers' data around K to do so. Just the tip of the band 
        is enough, since the cuts are 90 to 160 meV from the top valence band (TVB), which is given by WSe2. 
        DONE -> Used a range of 0.15 eV from VBM for the fitting.
                Used data of GG from both Gamma and M -> double fit with different mass for the 2 sides. 
                The fitting was done using a parabolic dispersion -k^2/2m.
    2-Compute CEM for cuts every 5 meV from TVB without interlayer coupling (IC). 
        Procedure:  we use for the energy the formula E(kx,ky) = mu - |k|^2/2m_M(cos^2(3/2 t)+m_M/m_G sen^2(3/2 t)) where m_G,m_M are the 
                    coefficients towards Gamma and M respectively, t=atan2(k_y/k_x). In this way we can interpolate between the two 
                    different slopes.
        To compute: - Fixed values: N = 6, k_pts = 201, range_K = 0.2 A^-1, e_cuts from 50meV to 200 meV below VBM, every 5 meV.
                    - Spread pars: Use Gaussian spreading. spread_k = 0.01, spread_E = 0.01 -> to see better side bands.
                    - energy cut: 50 meV below VBM

                    - Variable pars:    V: 0.001,0.0077(Louk val),0.02,0.03(Gamma val) (eV)
                                        phase: 0, -106°(Louk val), pi
                                        A_M: 50,60,70,80 (Angstrom)
                                        a,b: 0,0.5,1,5


#Change name files -> not absolute energy of the cut but distance from VBM.
                   -> change prcision of V in name from .3f to .4f (7.7 meV approx to 8 right now).


Procedure B:
    1-Use a single mass for the parabolic dispersion -> C6 symmetry without the interactions.
        Fit bilayer upper band around K (S3) -> get mass (already done somewhere).
    2-Compute CEM:
        Use for bilayer energy a 2x2 matrix with same dispersion up and down ?






