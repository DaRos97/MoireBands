# Monolayer fitting
We fit the monolayer bands with the 11 band model, starting from the DFT values of the tight-binding model. 

## Data structure
`.txt` files: 4 for each material, 2 cuts (K-G-Kp and K-M-Kp) and 2, 4 or 6 bands (top valence band with SOC, additional bands close to Gamma and M).
There are 6 bands only in the WSe2 because we consider the crossing at M and additional bands at Gamma.
The raw data has sometime NAN in the energy.
The symmetrized data has the average of the energy in plus/minus |k| when both points are not NAN, keep only one of the two is one is NAN and remove the k point if both are NAN.

The ARPES data at KGK and KMKp is adjusted to make the bands coincide at K in the upper band: bands on KMKp are shifted by -0.052 on WSe2 and +0.01 on WS2.
The final data file has both cuts on the same array, with |k| as if the two cuts were aligned.
We finally take a subset of points for the actual fitting.

For WS2 there are 2 cuts of given length with interpolated data on 2 bands.
For WSe2 there are 3 cuts of given length with intepolated data on 6 bands: Gamma-K, K-M', M'-M (M' is the point were we start to consider 4 bands for the crossing).
In the first cut for WSe2 depending on the k value we consider 6, 4 or 2 bands.

## Fit procedure
We start from the DFT values of the tb parameters (40+1(offset)+2(SOC)). 
We save the intermediate results.

The final function we minimize has:
- bands distance from experiment one -> standard chi2 with coefficient 1
- `Ppar`: distance of fitting parameters wrt to DFT ones with coefficient Ppar
- `Pbc`: orbital content at Gamma and K TVB and at K conduction band to be the DFT one with coefficient Pbc
- `Pdk`: three special points with higher importance: Gamma and K point of TVB and crossing point near M for bands 1 and 2. Coefficient Pdk
- `Pgap`: difference of band gap at Gamma, K and M wrt DFT one, to avoid completely irrealistic values, with coefficient Pgap

Bounds to parameters:
- rp: percentage difference for general orbitals
- rpz: percentage difference for z-orbitals (even)
- rpxy: percentage difference for xy-orbitals
- rl: percentage difference for SOC parameters

## Code structure
We can check the experimental data modifications and extractions in `data_visual.py`.
- `monolayer.py`
- `functions_monolayer.py`

### Analysis
- `select_best_result.py`
- `SortChi2.py`

### Utils:
- `ygg.sh`
- `bao.sh`
- `maf.sh`

### Cluster scripts
- `~*.sbatch` and `*.qsub` are for hpc
- `(q)job.sh` are for mafalda

# Results
The result used now is in big5.zip and saved in new_year result in Figures
There is another one, the best of big7.zip, which is better but has much larger parameter space and Ppar=0, so very distant from DFT.

## Mafalda
big from 1 to 7.











