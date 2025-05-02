import numpy as np
import functions as fs
import sys,os
from pathlib import Path
from scipy.linalg import eigh
import matplotlib.pyplot as plt

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
center, interlayer_type, step, N = fs.get_pars(int(sys.argv[1]))
title = "Center: "+center+", int type: "+interlayer_type+', step: '+"{:.3f}".format(step)+', N: '+str(N)
print(title)
if 0 and machine=='loc':
    exit()

#Fixed parameters
pars_V = (0.03,np.pi,0.015,np.pi)
DFT = False
a_Moire = 79.8
range_Kx = 1.2   #in A^-1, value of momentum around center
range_Ky = 0.5

#Moire parameters
n_cells = int(1+3*N*(N+1))
G_M = fs.get_Moire(a_Moire)
H_moire = [fs.H_moire(0,pars_V),fs.H_moire(1,pars_V)]
pars_moire = (N,pars_V,G_M,H_moire)

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
    offset[TMD] = temp[-3]
pars_monolayer = (hopping,epsilon,HSO,offset)
#Interlayer parameters
pars_interlayer = [interlayer_type,np.load(fs.get_pars_interlayer_fn(interlayer_type,DFT,machine))]
#BZ grid
pars_grid = (center, range_Kx, range_Ky, step)
grid = fs.get_grid(pars_grid)
kx_pts,ky_pts = grid[0].shape

#Compute energies and weights on grid
en_fn = fs.get_energies_fn(pars_grid,DFT,N,pars_V,a_Moire,interlayer_type,machine)
wg_fn = fs.get_weights_fn(pars_grid,DFT,N,pars_V,a_Moire,interlayer_type,machine)
ind_TVB = n_cells*28    #top valence band
ind_LVB = n_cells*24    #lowest considered VB
if not Path(en_fn).is_file() or not Path(wg_fn).is_file():
    print("Computing en,wg...")
    energies = np.zeros((kx_pts,ky_pts,ind_TVB-ind_LVB))
    weights = np.zeros((kx_pts,ky_pts,ind_TVB-ind_LVB))
    look_up = fs.lu_table(N)
    for i in tqdm(range(kx_pts*ky_pts)):
        x = i%kx_pts
        y = i//kx_pts
        K_i = np.array([grid[0][x,y],grid[1][x,y]])
        H_tot = fs.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
        energies[x,y,:],evecs = eigh(H_tot,subset_by_index=[ind_LVB,ind_TVB-1])           #Diagonalize to get eigenvalues and eigenvectors
        ab = np.absolute(evecs)**2
        weights[x,y,:] = np.sum(ab[:22,:ind_TVB-ind_LVB],axis=0) + np.sum(ab[22*n_cells:22*n_cells+22,:ind_TVB-ind_LVB],axis=0)
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
fig_dn = fs.get_fig_dn(pars_grid,DFT,N,pars_V,a_Moire,interlayer_type,pars_spread,machine)
if not Path(fig_dn).is_dir() and not machine=='loc':
    os.system('mkdir '+fig_dn)
#
e_cuts = np.linspace(-1.5,-1,11)

for en in e_cuts:
    cut_fn = fs.get_cut_fn(en,pars_grid,DFT,N,pars_V,a_Moire,interlayer_type,pars_spread,machine)
    if not Path(cut_fn).is_file():
        print("Computing energy cut ",en," ...")
        en_cut = np.zeros((kx_pts,ky_pts))
        G_E_tot = 1/((energies-en)**2+spread_E**2) if type_spread == 'Lorentz' else np.exp(-((energies-en)/spread_E)**2)
        Kx_list = np.arange(-range_Kx,range_Kx+step,step)
        Ky_list = np.arange(-range_Ky,range_Ky+step,step)
        if center == 'M':
            Ky_list += 2*np.pi/np.sqrt(3)/fs.dic_params_a_mono['WSe2']
        for i in tqdm(range(kx_pts*ky_pts)):
            x = i%kx_pts
            y = i//kx_pts
            G_K = fs.spread_fun_dic[type_spread](np.array([grid[0][x,y],grid[1][x,y]]),spread_k,Kx_list,Ky_list)
            for j in range(ind_TVB-ind_LVB):
                en_cut += abs(weights[x,y,j]**0.5)*G_E_tot[x,y,j]*G_K
        #Normalize in color scale
        en_cut = fs.normalize_cut(en_cut,(grid[0].shape))
        if save:
            np.save(cut_fn,en_cut)
    else:
        en_cut = np.load(cut_fn)
    
    #
    plt.figure(figsize=(18,10))
    plt.imshow(en_cut,cmap='gray')
    plt.title("En: "+"{:.4f}".format(en)+", "+title)
    plt.xlabel("$K_x(A^{-1})$",size=15)
    plt.ylabel("$K_y(A^{-1})$",size=15)
    plt.xticks([int(en_cut.shape[1]/10*i) for i in range(11)],["{:.2f}".format(-range_Kx+2*range_Kx/10*i) for i in range(11)],size=10)
    plt.yticks([int(en_cut.shape[0]/10*i) for i in range(11)],["{:.2f}".format(range_Ky-2*range_Ky/10*i) for i in range(11)],size=10)
    figname = fig_dn + "{:.4f}".format(en)+'.png'
    if 1 and machine == 'loc':
        plt.show()
    else:
        plt.savefig(figname)








