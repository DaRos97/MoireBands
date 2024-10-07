# Content
We fit the monolayer bands with the 11 band model, starting from the DFT values of the tight-binding model. 
The ARPES data at KGK and KMK is adjusted (offset in one of the 2) to make the bands coincide at K in the upper band.

In particular, we symmetrize the two sides of KGK and KMK and adjust the offset so that the symmetrized bands 
coincide at K.

We start from the DFT values and add a small (random) deviation in order to get to different minima of chi2. 
Each realization is numbered with an index given as parameter.
Usually the minimization does not finish so we save the intermediate results.

There are many ways to impose to the system to stay closer to the DFT values:
    1- After we have a bunch of results, we compute the distance of the solution to the initial DFT with another chi2
    and choose the closest one. 
    2- Add directly the `chi2_1` part (chi square distance of parameters) to the energy part (chi2_0) for different values of the coupling between the 2.

We should also discriminate the solutions based on the orbital content at G and K.

#Compute scripts
- `tight_binding.py`
- `functions.py`

#Copy from hpc: 'bash_code/'+
- `ygg.sh`
- `bao.sh`
- `maf.sh`

#Analyze results
- `select_best_result.py`


# Computed three pars
## Baobab
    - 1-500 bound 10.0 1.0 0.3
## Yggdrasil
## Mafalda
    - 1-50 bound 10.0 1.0 0.2
    - 1-50 bound 1.0 1.0 0.1
    - 1-50 bound 1.0 1.0 0.2
    - 1-50 bound 1.0 1.0 0.3
## Bamboo
    - 1-500 bound 10.0 1.0 0.2


# Computed four pars
## Yggdrasil
    - 1-100 bound 10.0 1.0 0.2 0.01
    - 1-100 bound 10.0 1.0 0.3 0.01
    - 1-100 bound 5.0 1.0 0.2 0.01
    - 1-100 bound 5.0 1.0 0.3 0.01
    - 1-20 bound [10.0,7.5,5,2.5] [0.5,1.0,1.5] [0.1,0.3,0.5] [0,0.01,0.1]
    - 1-50 bound 0 1 1 0
    - 1-50 bound 0 2 1 0
## Bamboo
    - 1-50 bound 10 0.5 0.1 0
    - 1-50 bound [10,0] [0.5,3] [0.01,2] 0
## Baobab
    - 1-20 bound [0-3,4] [0.5-3,11] [0.1-0.5,5] 0
    - 1-20 bound [0.1-0.9,5] [1-3,3] [0.1-0.5,3] 0
    # New name from here with 3 digits in spec args
    - 1-20 bound [0.05-0.15,6] [1-3,3] [0.1-0.5,3] 0



# Computing

## Baobab
## Yggdrasil
## Mafalda
## Bamboo



# Notes
Coefficient of `chi2_1` between 1 and 10 seems good. 
Maybe adjusting the bound on (e,t):
    - not just percentage but also either a constant term, maybe depending on 
      the absolute value of the term and/or the importance of the term (nn or nnn)

Just `t1_1111` and `t6_96` are changing sign

Can make code faster by reducing the number of k-points to consider -> `ind_reduced` parameter
The coefficient P changes with ths factor.

The top of lower band at K is always too small. And if you get close to it the orbital content at K changes to 0.

With `ind_reduced=7` the parameter P is below 1. 
With 0.1 get nice results but a bit far in parameter space, 0.3 is too big.
Problem is the SOC which saturates the bound every time














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
