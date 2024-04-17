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

# Computing
## Bao
    - 





# Computed
# Bao
    Old -> 0_:
    - N=5, px=5, sk=se=0.01, Gauss
    - DFT: True, False
    - Vg: 5, 10, 20, 30 meV
    - Vk: 1, 5, 7.7, 10, 15 meV
    - aM: 79.8, 70, 60, 50 A
    Saved images for DFT true and false, Vg 10,20,30, Vk 7.7, aM 79.8.

    New run
    - Interlayer: U1, C6, C3
    - N=5, px=5, sk=se=0.01, Gauss
    - DFT: True, False
    - Vg: 5, 10, 20, 30 meV
    - Vk: 1, 5, 7.7, 10, 15 meV
    - aM: 79.8, 70, 60, 50 A
    For DPG conference
    - C3, N=4, px=1, sk=se=0.01, Gauss
    - fit, (Vg,Vk,aM) = (0.03,0.0077,79.8)  -> index 228

# Computing
# Bao
    For DPG conference
    - C3, N=5, px=1, sk=se=0.01, Gauss
    - fit, (Vg,Vk,aM) = (0.03,0.0077,79.8)



# To check

HSO
