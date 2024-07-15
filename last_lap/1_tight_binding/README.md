# Content
We fit the monolayer bands with the 11 band model, starting from the DFT values. 
The data at KGK and KMK is adjusted (offset in one of the 2) to make the bands coincide at K.
We change: material, with/without SO in minimization and the range of the parameters.
The offset is chosen initially by eye comparison and then has a larger window of change.

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
Here can decide to plot either temp/ or results/ fit parameters.

#DFT distance
- `distance_dft.py`
Computes a table with difference of resulting pars from DFT ones, with percentage.
Also here can decide which result to use for the table.

#Orbital content
- `orbital_content.py`
Computes the relevant orbital content close to Gamma and K.

# Computed
##Maf
    - Range 0.1 to 1, WSe2 and WS2, SO fixed and not.
    - Range 0.1 to 1, WSe2 and WS2, SO fixed and not with offset not constrained

# Computing
##Bao
    - fixed range 0.1, gamma=logrange(2,4,20)
##Maf
    - fixed range 0.1, gamma=logrange(2,4,20)


# Best fitting
    WSe2 -> 0.4, fixed_SO, constrained offset
    WS2 -> 0.2, fixed_SO , constrained offset
