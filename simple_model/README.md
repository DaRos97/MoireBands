# Interlayer hopping and Moirè potential

Around Gamma:
- We take a parabolic dispersion for each band: *-k^2/2m*  --> one parameter for each band
- We take an energy offset *c* for the lower band
- We include an interlayer hopping of the form *-a(1-bk^2)*  --> hopping stronger at k=0 since the d_z orbital is stronger. Parameter b defines how quickly this decreases

In file "interlayer_1.py" we take the bilayer data from experiment and fit it with the Hamiltonian

H = (   -k^2/2m_1   -a(1-bk^2)      )
    (   -a(1-bk^2)  -k^2/2m_2-c     )

Like this we find the values of m_1,m_2, c and of the interlayer hopping.

In file "interlayer_2.py" we go further and use these values together with the Moirè coupling. 
The Hamiltonian now is divided in 4 blocks: H = (A B \\ C D), B=C

A and D represent the single layers. We take a number of mini-BZs, which are on the diagonal, and are coupled to each other by the Moirè potential,
which reads V=|V|e^{i\phi} with sign +/- in the exponent depending on the reciprocal Moirè lattice vector connecting the two mini-BZs. 

B and C give the interlayer hopping, which is k dependent and acts only within the same mini-BZ --> diagonal matrices.

We diagonalize the big H for the path K-G-K' and compute the ARPES overlap to get a plot similar to the experiment. We want to see for which values of 
the Moirè potential we see the features of the experiment, which are mainly two strong Moirè replicas for the upper layer. 

We can think of a fit for this step as well, just in Moirè potential amplitude and phase.

In file "interlayer_3.py" we fit the Moirè potential with the data.

We care only about the Moirè of the top band. We want to fit the data with the Model we have. In order to do that we compute a final image (with Lorentz spreading) 
with exactly the same size as the data (same number of pixels) --> need to fix the grid accordingly. Then we do a chi^2 minimization by comparing each produced picture 
pixel by pixel. 

We do this at Gamma with the model described above, and the same at K using another image and a different model --> ask Louk.

# Fitting at K-point

Using the image from experiments in the cut Gamma-K-M we fit first the upper band (neglect lower bands and spin orbit oupling) with a parabolic dispersion k^2/2m --> get m.

Then we insert the moirè potential and fit the image pixel by pixel with the data as done for the Gamma point in order to get the Moirè potential at K.

# Data

The experiment image is on the cut K-G-K'. The inset "KGK_WSe2onWS2_forDario.png" goes from -0.5 to 0.5 A^-1 and in energy from -0.5 to -1.7 eV.

# Others

We want here to see the 3-fold rotation symmetry instead of a normal 6-fold we would expect. We need to add the spin-orbit coupling, but different depending if we go
towards K or K' from Gamma. Then, we want to do a spin-projected ARPES weight and see what happens.

Spin orbit coupling comes from interaction between angular momentum and spin. 
