import numpy as np
import functions as fs
import sys,os
from pathlib import Path
from scipy.linalg import eigh

machine = fs.get_machine(os.getcwd())
if machine=='loc':
    from tqdm import tqdm
    save = False
else:
    tqdm = fs.tqdm
    save = True

"""
Here we compute CEMs around G and around K.
"""
#Variable parameters
center, DFT, pars_V, a_Moire = fs.get_pars(int(sys.argv[1]))
title = "Center: "+center+", DFT: "+str(DFT)+", pars_V: "+fs.get_list_fn(pars_V)+", a_Moire: "+str(a_Moire)
print(title)
#Moire parameters
N = 0                               #####################
n_cells = int(1+3*N*(N+1))
G_M = fs.get_Moire(a_Moire)
pars_moire = (N,pars_V,G_M)
#Monolayer parameters
hopping = {}
epsilon = {}
HSO = {}
offset = {}
for TMD in fs.materials:
    temp = np.load(fs.get_pars_mono_fn(TMD,machine,DFT))
    hopping[TMD] = fs.find_t(temp)
    epsilon[TMD] = fs.find_e(temp)
    HSO[TMD] = fs.find_HSO(temp)
    offset[TMD] = temp[-1]
pars_monolayer = (hopping,epsilon,HSO,offset)
#Interlayer parameters
pars_interlayer = np.load(fs.get_pars_interlayer_fn(machine,DFT))
#BZ grid
range_K = 0.3   #in A^-1, value of momentum around center
k_pts = 51
pars_grid = (center, range_K, k_pts)

#Compute energies and weights on grid
en_fn = fs.get_energies_fn(pars_grid,DFT,N,pars_V,a_Moire,machine)
wg_fn = fs.get_weights_fn(pars_grid,DFT,N,pars_V,a_Moire,machine)
ind_TVB = n_cells*28    #top valence band
ind_LVB = n_cells*22    #lowest considered VB
if not Path(en_fn).is_file() or not Path(wg_fn).is_file():
    print("Computing en,wg...")
    grid = fs.get_grid(pars_grid)
    energies = np.zeros((k_pts,k_pts,ind_TVB-ind_LVB))
    weights = np.zeros((k_pts,k_pts,ind_TVB-ind_LVB))
    look_up = fs.lu_table(N)
    for i in tqdm(range(k_pts**2)):
        x = i%k_pts
        y = i//k_pts
        K_i = np.array([grid[0][x,y],grid[1][x,y]])
        H_tot = fs.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
        energies[x,y,:],evecs = eigh(H_tot,subset_by_index=[ind_LVB,ind_TVB-1])           #Diagonalize to get eigenvalues and eigenvectors
        for e in range(ind_TVB-ind_LVB):
            for l in range(2):
                for d in range(22):
                    weights[x,y,e] += np.abs(evecs[d+22*n_cells*l,e])**2
    if save:
        np.save(en_fn,energies)
        np.save(wg_fn,weights)
else:
    energies = np.load(en_fn)
    weights = np.load(wg_fn)


#Compute spread and final picture
spread_k = 0.01
spread_E = 0.01
type_spread = 'Gauss'
pars_spread = (spread_k,spread_E,type_spread)
#
cut_dn = fs.get_cut_dn(pars_grid,DFT,N,pars_V,a_Moire,machine)
if not Path(cut_dn).is_dir() and not machine=='loc':
    os.system('mkdir '+cut_dn)
#
max_E = np.max(energies)
e_cuts = [max_E-0.005*i for i in range(1,11)]
for en in e_cuts:
    cut_fn = fs.get_cut_fn(max_E-en,pars_grid,DFT,N,pars_V,a_Moire,pars_spread,machine)
    if not Path(cut_fn).is_file():
        print("Computing energy cut ",max_E-en," ...")
        en_cut = np.zeros((k_pts,k_pts))
        G_E_tot = 1/((energies-en)**2+spread_E**2) if type_spread == 'Lorentz' else np.exp(-((energies-en)/spread_E)**2)
        Kx_list = np.linspace(-range_K,range_K,k_pts) 
        if center == 'K':
            Kx_list += 4/3*np.pi/fs.dic_params_a_mono['WSe2']
        Ky_list = np.linspace(-range_K,range_K,k_pts)
        grid = fs.get_grid(pars_grid)
        for i in tqdm(range(k_pts**2)):
            x = i%k_pts
            y = i//k_pts
            G_K = fs.spread_fun_dic[type_spread](np.array([grid[0][x,y],grid[1][x,y]]),spread_k,Kx_list,Ky_list)
            for j in range(ind_TVB-ind_LVB):
                en_cut += abs(weights[x,y,j]**0.5)*G_E_tot[x,y,j]*G_K
        #Normalize in color scale
        en_cut = fs.normalize_cut(en_cut,pars_grid)
        if save:
            np.save(cut_fn,en_cut)
    else:
        en_cut = np.load(cut_fn)

    if 1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18,10))
        plt.imshow(en_cut,cmap='gray')
        plt.title("En: "+"{:.4f}".format(max_E-en)+", "+title)
        Kxm = -range_K if center == 'G' else 4/3*np.pi/fs.dic_params_a_mono['WSe2']-range_k
        KxM = range_K if center == 'G' else 4/3*np.pi/fs.dic_params_a_mono['WSe2']+range_k
        Kxc = 0 if center == 'G' else 4/3*np.pi/fs.dic_params_a_mono['WSe2']
        plt.xticks([0,k_pts//2,k_pts],["{:.2f}".format(Kxm),"{:.2f}".format(Kxc),"{:.2f}".format(KxM)])
        plt.yticks([0,k_pts//2,k_pts],["{:.2f}".format(range_K),"{:.2f}".format(0),"{:.2f}".format(-range_K)])
        plt.xlabel("$K_x(A^{-1})$",size=15)
        plt.ylabel("$K_y(A^{-1})$",size=15)
        if machine == 'loc':
            plt.show()
        else:
            plt.savefig(fs.get_fig_fn(max_E-en,pars_grid,DFT,N,pars_V,a_Moire,pars_spread,machine))








