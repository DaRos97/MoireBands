# Monolayer fitting
We fit the monolayer bands with the 11 band model, starting from the DFT values of the tight-binding model. 

## Data structure

`.txt` files: 4 for each material, 2 cuts (K-G-Kp and K-M-Kp) and 2 or 4 bands (top valence band with SOC).
There are 4 bands only in the WSe2 because we consider the crossing at M.
The raw data has sometime NAN in the energy.
The symmetrized data has the average of the energy in plus/minus |k| when both points are not NAN, keep only one of the two is one is NAN and remove the k point if both are NAN.

The ARPES data at KGK and KMKp is adjusted to make the bands coincide at K in the upper band: bands on KMKp are shifted by -0.052 on WSe2 and +0.01 on WS2.
The final data file has both cuts on the same array, with |k| as if the two cuts were aligned.
We finally take a subset of points for the actual fitting.

For WS2 there are 2 cuts of given length with interpolated data on 2 bands.
For WSe2 there are 3 cuts of given length with intepolated data on 4 bands. On the first 2 cuts only the first 2 bands are non-NAN.

## Fit procedure
We keep the SOC to be the DFT one and vary the other parameters.

We start from the DFT values of the remaining tb parameters (40+1(offset)). 
Usually the minimization does not finish so we save the intermediate results.

The final function we minimize has:
- bands distance from experiment one -> standard chi2 with coefficient 1
- distance of fitting parameters wrt to DFT ones with coefficient P -> best at 0.05 (depends on number of considered momentum points, here 1/13)
- distance of Gamma and K point of the two bands from the experiment with coefficient Pdk=20
- orbital content at Gamma and K to be the DFT one with coefficient Pbc=10

We add `rp` as the percentage max distance from the DT parameters -> best at 50%.


## Code structure
We can check the experimental data modifications and extractions in `data_visual.py`.
- `monolayer.py`
- `functions_monolayer.py`

### Analysis
- `select_best_result.py`

### Utils:
- `ygg.sh`
- `bao.sh`
- `maf.sh`

### Cluster scripts
- `~*.sbatch` and `*.qsub` are for hpc
- `(h)job.sh` are for mafalda

# Results
## Mafalda
    TMD=WSe2,WS2
    lP=[0.01,0.05,0.1,0.5,1,5]
    lrp = [0.2,0.5,1,1.5,2]
    lrl = 0
    ind_reduced = 13
    Pbc = 10
    Pdk = 20
    BEST: P=0.05, rp=50% for both TMD












