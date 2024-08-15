# Content
We fit the monolayer bands with the 11 band model, starting from the DFT values. 
The ARPES data at KGK and KMK is adjusted (offset in one of the 2) to make the bands coincide at K. In particular,
we symmetrize the two sides of KGK and KMK and adjust the offset so that the symmetrized bands coincide at K.

We start from the DFT values and add a small (random) deviation in order to get to different minima of chi2. 
Each realization is numbered with an index given as parameter.
Usually the minimization does not finish so we save the intermediate results.

After we have a bunch of results, we compute the distance of the solution to the initial DFT with another chi2
and choose the closest one. 
We should also discriminate the solutions based on the orbital content at G and K.

#Compute files
- `tight_binding.py`
- `functions.py`
- `parameters.py`

#Copy from hpc
- `ygg.sh`
- `bao.sh`
- `maf.sh`
Copies both temp/ and resutls/ from hpc to local and remove old results.

#Plot
- `Plot_final.py`

#DFT distance
- `distance_dft.py`
Computes a table with difference of resulting pars from DFT ones, with percentage.
Also here can decide which result to use for the table.

#Orbital content
- `orbital_content.py`
Computes the relevant orbital content close to Gamma and K.

# Computed
# Computing

# OLD RESULTS -> range of parameters and slow computation
# Computed
##Maf
    - Range 0.1 to 1, WSe2 and WS2, SO fixed and not.
    - Range 0.1 to 1, WSe2 and WS2, SO fixed and not with offset not constrained

##Bao
    - fixed range 0.1, gamma=logrange(2,4,20)
# Computing
##Maf
    - fixed range 0.1, gamma=logrange(1,4,20), SOmin


# Best fitting
    WSe2 -> 0.4, fixed_SO, constrained offset
    WS2 -> 0.2, fixed_SO , constrained offset
