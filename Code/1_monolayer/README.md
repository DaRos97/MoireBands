# Content
We fit the monolayer bands with the 11 band model, starting from the DFT values of the tight-binding model. 
The ARPES data at KGK and KMK is adjusted (offset in one of the 2) to make the bands coincide at K in the upper band.

In particular, in the experimental data we symmetrize the two sides of KGK and KMK and adjust the offset so that the symmetrized bands coincide at K.

We keep the SOC to be the DFT one and vary the other parameters.

We start from the DFT values of the remaining tb parameters (40) and add a small (random) deviation in order to get to different minima of chi2. 
Each realization is numbered with an index given as parameter. Usually the minimization does not finish so we save the intermediate results.

The final function we minimize has:
- bands distance from exp one -> standard chi2 with coefficient 1
- distance of fitting parameters wrt to DFT ones with coefficient P -> best at 0.05 (depends on number of considered momentum points, here 1/13)
- distance of Gamma and K point of the two bands from the experiment with coefficient Pdk=20
- orbital content at Gamma and K to be the DFT one with coefficient Pbc=10

We add `rp` as the percentage max distance from the DT parameters -> best at 50%.

#Compute scripts
- `monolayer.py`
- `functions_monolayer.py`

#Analyze results
- `select_best_result.py`

#Copy from hpc: 'bash_code/'+
- `ygg.sh`
- `bao.sh`
- `maf.sh`

#Job scripts
- `~*.sbatch` and `*.qsub` are for hpc
- `(h)job.sh` are for mafalda

# Computed
## Mafalda
    TMD=WSe2,WS2
    lP=[0.01,0.05,0.1,0.5,1,5]
    lrp = [0.2,0.5,1,1.5,2]
    lrl = 0
    ind_reduced = 13
    Pbc = 10
    Pdk = 20
    BEST: P=0.05, rp=50% for both TMD












