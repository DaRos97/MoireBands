Here we compute the final CEM to compare to those of S11. The procedue is the same as for step 3
but the calculation is longer due to the fact that we need to evaluate a grid in k space instead
of a line. 
For this step we use some specific prameters derived in step 3, in particular the Moire ones: 
(V,phi) at gamma and K.

# Code
get_1_2_3.py    imports the parameters from previopus steps
cem.py & functions.py       compute the cem

# Computed
##Bao
    - rgK 0.3, kpts 101, N=3, Gauss, sk=se=0.01, DFT, Vg 0.01, Vk 0.0077,
      Gamma and K, aM 50,60,70,79.8
    

# Computing
##Bao
    - Gamma, rgK 0.3, kpts 101, N=4, Gauss, sk=se=0.01, DFT, Vg 0.01, Vk 0.0077,
        aM 50,60,70,79.8
    - rgK 0.3, kpts 101, N=3, Gauss, sk=se=0.01, Vk 0.0077, aM 79.8
        Gamma and K, DFT and min, Vg 0.01, 0.02, 0.03
