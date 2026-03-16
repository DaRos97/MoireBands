# Monolayer fitting
We fit the monolayer bands with the 11 band model (plus SOC), starting from the DFT-derived values of the tight-binding model. 

## ARPES data structure
All files `.txt` of data from ARPES are in the `$HOME/Inputs/` folder.
There are two cuts: `KpGK` and `KpMK`, each having six and four bands, respectively, for each material.
The data is processed in three steps:
    - 1: The `raw` data: contains the energy and momenta from the ARPES fitting of intensities. This contains some NAN in the energy.
    - 2: The `symmetrized` data: the plus and minus k values are averaged (where present, some lower bands were fitted only for positive/negative momenta).
    - 3: The `merged` data: the two cuts are put together (adjusting with an offset to make them coincide at K: 52 meV in WSe2 and 10 meV in WS2).
        In this step I also chose a number of points I want for the merged data and interpolate all the bands to have this equi-distanced set.
        The merged data is the one used in the fitting of the tight-binding parameters.

## Fit procedure
We start from the DFT-derived values of the tb parameters (40+1(offset)+2(SOC)). 
We save the intermediate results.

The final function we minimize has:
- `chi2` of bands distance from experiment one -> standard chi2 with coefficient 1
- Constraint `K1`: distance of fitting parameters wrt to DFT ones with coefficient Ppar
- Constraint `K2`: orbital content at Gamma, K and M TVB to be OPO, IPO and IPO respectively
- Constraint `K2b`: orbital content at K BCB to be OPO
- Constraint `K3`: minimum of conduction band at K
- Constraint `K4`: difference of band gap at K wrt DFT one
- Constraint `K5`: special points with higher importance for the fit, namely G, K, min{K-M}, M of TVB

Bounds to parameters:
- rp: percentage difference for general orbitals
- rpz: percentage difference for z-orbitals (even)
- rpxy: percentage difference for xy-orbitals
- rl: percentage difference for SOC parameters

## Code structure
`visualize_ARPES_data.py` to see the experimental data for each TMD how is manipulated: raw, symmetrized and merged. All the manipulation is performed in `$HOME/CORE_functions.py`.
`main.py` is the script for the fitting of the model.
`utils.py` where the actual chi2, constraints, plotting and parameters are defined.
Usage: `python main.py arg1 arg2` with arg1 in {WS2, WSe2} and arg2 index. 
The index parameter is used to chose the set of bounds/contraint constants to use in the fit.
Each set leads to a best solution at the end of the minimization saved in a file `Data/temp_TMD_*args/temp_{chi2}.py`.
The whole group of sets is compressed in a set of solutions `setsTMD/set{i}.zip` using the `./zipBash.sh TMD i` bash script.
The set is moved to the local machine in `Data/setsTMD/`, where it is unzipped.
The analysys is carried out with `python sortResults.py TMD` which sorts the results with their final value of the minimizing function.
It sequentially plots bands, orbitals and parameters of the final solution.

## Results










