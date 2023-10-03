# G1 - Preliminary

We start by cutting the initial image given by GM, in the window -0.5/-1.7 eV and -0.5/0.5 A^-1.

# G2 - Interlayer hopping

We first extract the darkest points to see the two bands. Then we fit them with the simplified model below.

Around Gamma:
- We take a parabolic dispersion for each band: *-k^2/2m*  --> one parameter for each band
- We take an energy offset *c* for the lower band
- We include an interlayer hopping of the form *-a(1-bk^2)*  --> hopping stronger at k=0 since the d_z orbital is stronger. Parameter b defines how quickly this decreases
- An overall chemical potential *mu*

The Hamiltonian is

H = (   -k^2/2m_1+mu   -a(1-bk^2)      )
    (   -a(1-bk^2)  -k^2/2m_2-c+mu     )

Like this we find the values of m_1,m_2, c and of the interlayer hopping.

# G3 - Moire fitting

Here we first cut the image to consider just the upper band. Then we take the parameters obtained in G2 and we construct Moirè Hamiltonians of the form: 

The Hamiltonian now is divided in 4 blocks: H = (A B \\ C D), B=C

A and D represent the single layers. We take a number of mini-BZs, which are on the diagonal, and are coupled to each other by the Moirè potential,
which reads V=|V|e^{i\phi} with sign +/- in the exponent depending on the reciprocal Moirè lattice vector connecting the two mini-BZs. 

B and C give the interlayer hopping, which is k dependent and acts only within the same mini-BZ --> diagonal matrices.

We diagonalize the big H for the path K-G-K' and compute the ARPES overlap to get a plot similar to the experiment. We want to see for which values of 
the Moirè potential we see the features of the experiment, which are mainly two strong Moirè replicas for the upper layer. 

We construct png images with various values of V, phase and spread in E and K direction. Once we obtain a reasonable agreement by eye of what the parameters should be, 
we start a minimization routine where we minimize the difference between the constructed image and the experimental one.

# K1 - Preliminary

The same at K using another image and a different model. We have three light polarizations: CL,LH,LV. In K1 we just remove the border of empty and black pixels.

# K2 - Interlayer hopping

We use the left part (G-K) of CL picture to fit just the upper band with a parabolic dispersion of the type

H = -k^2/(2m) + mu

We consider only a small portion of k (~ 0.1 A^-1) close to the K point so that it actually looks like a parabula.

# K3 - Moire fitting

We use the LH light for this step, still in the left part (G-K cut).

# Fitting at K-point

Using the image from experiments in the cut Gamma-K-M we fit first the upper band (neglect lower bands and spin orbit oupling) with a parabolic dispersion k^2/2m --> get m.

Then we insert the moirè potential and fit the image pixel by pixel with the data as done for the Gamma point in order to get the Moirè potential at K.

# Data

The experiment image is on the cut K-G-K'. The inset "KGK_WSe2onWS2_forDario.png" goes from -0.5 to 0.5 A^-1 and in energy from -0.5 to -1.7 eV.

# Others

We want here to see the 3-fold rotation symmetry instead of a normal 6-fold we would expect. We need to add the spin-orbit coupling, but different depending if we go
towards K or K' from Gamma. Then, we want to do a spin-projected ARPES weight and see what happens.

Spin orbit coupling comes from interaction between angular momentum and spin. 
