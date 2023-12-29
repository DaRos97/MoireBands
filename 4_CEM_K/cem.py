import numpy as np
import functions as fs
from PIL import Image
from pathlib import Path
import os

cluster = False if os.getcwd()[6:11]=='dario' else True
if not cluster:
    from tqdm import tqdm
else:
    tqdm = fs.tqdm

#We compute first the energies and weights of all the k-pts and store them
#Hopt
Hopt_WS2 = np.load(fs.Hopt_filename('WS2','up','2',cluster))
Hopt_WSe2 = np.load(fs.Hopt_filename('WSe2','up','2',cluster))
pars_H = (Hopt_WS2,Hopt_WSe2)
#Moire
N = 5   #Usual number of mini-BZ circles around central one
V = 0.0077
phase = -106/180*np.pi
A_M = 79.8      #Moire length -> Angstrom
pars_moire = (N,V,phase,A_M)
#Grid
range_K = 0.2   #in A^-1, value of momentum around 0 (we take K the center of coordinates)
k_pts = 201     #number of k-pts in each direction, in the range specified
pars_grid = (range_K, k_pts)
#Interlayer
a = 0
b = 0
c = 0
pars_interlayer = (a,b,c)

en_filename = fs.energies_filename(pars_moire,pars_grid,pars_interlayer,cluster)
wg_filename = fs.weights_filename(pars_moire,pars_grid,pars_interlayer,cluster)
if not Path(en_filename).is_file() or not Path(wg_filename).is_file():
    print("Computing en,wg...")
    grid = fs.get_grid(pars_grid)
    n_cells = int(1+3*N*(N+1))        #Index of higher valence band 
    energies = np.zeros((k_pts,k_pts,2*n_cells))
    weights = np.zeros((k_pts,k_pts,2*n_cells))
    G_M = fs.get_Moire(A_M)
    look_up = fs.lu_table(N,G_M)
    for i in tqdm(range(k_pts**2)):
        x = i%k_pts
        y = i//k_pts
        K = np.array([grid[0][x,y],grid[1][x,y]])
        H_tot = fs.big_H(K,look_up,pars_H,pars_moire,pars_grid,pars_interlayer,G_M)
        energies[x,y,:],evecs = np.linalg.eigh(H_tot)           #Diagonalize to get eigenvalues and eigenvectors
        for e in range(2*n_cells):
            for l in range(2):
                weights[x,y,e] += np.abs(evecs[n_cells*l,e])**2
    np.save(en_filename,energies)
    np.save(wg_filename,weights)
else:
    energies = np.load(en_filename)
    weights = np.load(wg_filename)

#We now compute the CEMs 
#Spread
spread_k = 0.01
spread_E = 0.05
type_spread = 'Gauss'
pars_spread = (spread_k,spread_E,type_spread)
#Energy cuts
max_E = np.max(energies)
e_cuts = [max_E-0.05 - 0.005*n for n in range(30)]

for en in e_cuts:
    cut_filename = fs.energy_cut_filename(en,pars_moire,pars_grid,pars_interlayer,pars_spread,cluster)
    if not Path(cut_filename).is_file():
        print("Computing en_cut...")
        en_cut = np.zeros((k_pts,k_pts))
        n_cells = int(1+3*N*(N+1))
        G_E_tot = 1/((energies-en)**2+spread_E**2) if type_spread == 'Lorentz' else np.exp(-((energies-en)/spread_E)**2)
        K_list = np.linspace(-range_K,range_K,k_pts)
        grid = fs.get_grid(pars_grid)
        for i in tqdm(range(k_pts**2)):
            x = i%k_pts
            y = i//k_pts
            G_K = fs.spread_fun_dic[type_spread](np.array([grid[0][x,y],grid[1][x,y]]),spread_k,K_list)
            for j in range(2*n_cells):
                en_cut += abs(weights[x,y,j]**0.5)*G_E_tot[x,y,j]*G_K
        #Normalize in color scale
        en_cut = fs.normalize_cut(en_cut,pars_grid)
        np.save(cut_filename,en_cut)
    else:
        en_cut = np.load(cut_filename)
    #Final image
    fs.compute_image_CEM(en_cut,en,max_E,pars_moire,pars_grid,pars_interlayer,pars_spread,cluster)






















