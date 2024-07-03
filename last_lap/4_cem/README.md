Here we compute the final CEM to compare to those of S11. The procedue is the same as for step 3
but the calculation is longer due to the fact that we need to evaluate a grid in k space instead
of a line. 
For this step we use some specific prameters derived in step 3, in particular the Moire ones: 
(V,phi) at gamma and K.

# Code
get_1_2.py    imports the parameters from previopus steps
cem.py & functions.py       compute the cem

# Computed
##Maf
    First try: 
        - V=(0.03,pi,0.015,pi), fit, aM=79.8, rK=(1.2,0.5), spread=(0.01,0.01,Gauss)
        - E cuts: (-1.5,-1,11)
        - cenetr: G-M, int: C6-C3, step: 0.05-0.04-0.03-0.02-0.01, N: 2-3-4
        ERROR in cut_fn -> only first energy cut at -1.5 is computed. Energies and weigths are ok, so can be used to compute other cuts.
    

# Computing
##Maf
    - V=(0.03,pi,0.015,pi), fit, aM=79.8, rK=(1.2,0.5), spread=(0.01,0.01,Gauss)
    - E cuts: (-1.5,-1,11)
    - cenetr: G-M, int: C6-C3, step: 0.03-0.02-0.01-0.005, N: 0-1-2-3-4
