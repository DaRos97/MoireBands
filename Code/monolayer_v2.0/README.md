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
All results were computed on the `mafalda` DQMP cluster.
The final chosen values are saved in `$HOME/Inputs/` with signature `tb_TMD_B:$1_K:$K1_$K2_$K2b_$K3_$K4_$K5.npy` in order to be used in the bilayer analysis.

### WSe2
`set2.zip`
- Ks: 1e-3, 1e-3, 0, 0, 0.1, 5      ->  min of conduction not at K and orbitals of K mostly p_xy, maybe beter d_xy
`set3.zip`
- Ks: 1e-3, 1e-3, 0, 1, 0.1, 5      ->  little pz at M, 4 saturated par bounds
`set4.zip` - B:5
- Ks: 0.001, 0.005, 0, 1, 0.1, 10 ->
`set5.zip` - B:5
- Ks: 0.00005, 0.1 , 0, 1, 0.1, 10 -> orbM=0.0049, slightly closer SOC b/w G and K -> BEST
- Ks: 0.00005, 0.05, 0, 1, 0.1, 10 -> orbM=0.0118
- Ks: 0.005,   0.1 , 0, 1, 0.1, 10 -> orbM=0.0039, closer SOC b/w G and K

### WS2
`set1.zip`
- Ks: 1e-3, 1e-2, 0, 1, 1, 10      ->  pz at M, 4 saturated par bounds
- Ks: 1e-3, 1e-2, 0, 1, 0.1, 10      ->  4 saturated par bounds
`set4.zip` - B:5
- Ks: 0.001, 0.01, 0, 1, 0.1, 10 -> not so pretty, orbM=0.059
`set5.zip` - B:5
- Ks: 0.00001, 0.05, 0, 1, 0.1, 10 -> not so pretty, orbM=0.015
- Ks: 0.00005, 0.05, 0, 1, 0.5, 5  -> large Ps, orbM=0.005
- Ks: 0.00005, 0.1 , 0, 1, 0.5, 5  -> large Ps, orbM=0.003
- Ks: 0.0001,  0.1 , 0, 1, 0.1, 5  -> large Ps (3 sta), orbM=0.005 -> BEST










