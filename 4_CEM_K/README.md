# What's what

Here we compute CEM around K. We do it in two different ways: with and without interlayer.
Why?
Because in case of 0° rotation, the two layers have K on top of K (referred to as parallel 'P') but in case of 60° rotation K is on 
top of K' (anti-parallel 'AP'). In both cases we expect C3 symmetry because of the BZ shape (K to M or to Gamma). For P there is 
in principle more interlayer coupling since the spin is locked to the valley and the two valleys have the same spin. Instead for AP
the interlayer should be reduced.

In this code we use the parabolic single orbital ('simple') model, with and without interlayer coupling.

For the bilayer we use a Hamiltonian of the form:
H = (E up           t(k)            )       with t(k) to be defined and a diagonal energy shift mu. and E up/down defined below.
    (t*(k)          E down - c      )

In the P case we fit the image using a and b. In the AP case instead we put a=b=0.

S11 seems to have C6 symmetry, with intensities probably modulated by the matrix elements. This could be due to weak interlayer coupling -> AP case.
S3 seems to have a more clear C3 symmetry. This could be due to more relevant interlayer coupling -> P case.

STEPS:
1-Extract band parameters for WSe2 and WS2, respectively. We take the monolayers' data around K to do so. Just the tip of the band 
    is enough, since the cuts are 90 to 160 meV from the top valence band (TVB), which is given by WSe2. 
    DONE -> Used a range of 0.15 eV from VBM for the fitting.
            Used data of GG from both Gamma and M -> double fit with different mass for the 2 sides. 
            The fitting was done using a parabolic dispersion -k^2/2m.
2-Compute CEM for cuts every 5 meV from TVB without interlayer coupling (IC). 
    Procedure:  we use for the energy the formula E(kx,ky) = mu - |k|^2/2m_2(cos^2(3/2 t)+m_2/m_1sen^2(3/2 t)) where m_G,m_M are the 
                coefficients towards Gamma and M respectively, t=atan2(k_y/k_x). In this way we can interpolate between the two 
                different slopes.
    To compute: - Fixed values: N = 5, k_pts = 201, range_K = 0.2 A^-1, e_cuts from 50meV to 200 meV below VBM, every 5 meV.
                - Spread pars: Use Gaussian spreading. spread_k = 0.01, spread_E = 0.01 -> to see better side bands.
                - Variable pars: (V,phase) from Louk paper (7.7 meV,-106˚). A_M = 79.8 A. 
    Compute:    - Basic above -> running Ygg
                - Basic with t(k) = -a with a=1 (from fitting around Gamma.
                - Basic with t(k) = -a with a=0.1 (from fitting around Gamma.
                - Basic with t(k) = -a with a=0.01 (from fitting around Gamma.
3-Add IC in the form of:    
        a constant: t(k) = -a -> compute some cases (see value of a around Gamma for a starting value);
        a constant plus a squared k dependence which is 0 at k: t(k) = -a(1-b*k^2) -> compute some cases (again look at the value of a and b from Gamma for reference);

