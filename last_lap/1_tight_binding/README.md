#Compute files
- tight_binding.py
- functions.py
- parameters.py

#Copy from hpc
- ygg.sh
- bao.sh
Copies both temp/ and resutls/ from hpc to local and remove old results.

#Plot
- Plot_final.py
Here can decide to plot either temp/ or results/ fit parameters.

#DFT distance
- distance_dft.py
Computes a table with difference of resulting pars from DFT ones, with percentage.
Also here can decide which result to use for the table.

#Orbital content
- orbital_content.py
Computes the relevant orbital content close to Gamma and K.

# Computed
##Ygg:
##Bao
    0 - Computing same just for WS2 and just KGK+KMKp after adjusting offset
    - Range 0.1 to 1, WSe2 and WS2, KGK+KMKp and KGK with adjusted offset.

# Computing

# Local
    0 - old results
    1 - newish results
    _ - latest results
# Best fitting
## Old:
    WSe2 -> 0.8, KGK
    WS2  -> 0.5, KGK-KMKp
## Final:
    WSe2 -> 0.5, KGK+KMKp 
    WS2  -> 0.3, KGK+KMKp (or 1 KGK+KMKp for result fitting better)
