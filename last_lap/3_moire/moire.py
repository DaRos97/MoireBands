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
Here we compute the full image S11.
We need:
    - monolayer parameters
    - interlayer parameters
    - moire copies -> V and phi around G and K
"""

#Moire parameters
N = 0                               #####################
n_cells = int(1+3*N*(N+1))
"""
Moirè potential of bilayer
Different at Gamma (d_z^2 orbital) -> first two parameters, and K (d_xy orbitals) -> last two parameters
Gamma point values from paper "G valley TMD moirè bands" (first in eV, second in radiants)
K point values from Louk's paper (first in eV, second in radiants)
"""
DFT, pars_V, a_Moire = fs.get_pars(int(sys.argv[1]))
title = "DFT: "+str(DFT)+", pars_V: "+fs.get_list_fn(pars_V)+", a_Moire: "+str(a_Moire)
print(title)
#
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
#Extract S11 image
S11_fn = fs.get_S11_fn(machine)
K = 4/3*np.pi/fs.dic_params_a_mono['WSe2']
EM = -0.5
Em = -2.5
bounds = (K,EM,Em)
exp_pic = fs.extract_png(S11_fn,[-K,K,EM,Em])
pixel_factor = 5            ###################################
#BZ cut parameters
cut = 'KGK'
k_pts = exp_pic.shape[1]//pixel_factor
K_list = fs.get_K(cut,k_pts)
K_scalar = np.zeros(k_pts)
for i in range(k_pts):
    K_scalar[i] = np.linalg.norm(K_list[i])

Compute energies and weights along KGK
en_fn = fs.get_energies_fn(DFT,N,pars_V,pixel_factor,a_Moire,machine)
wg_fn = fs.get_weights_fn(DFT,N,pars_V,pixel_factor,a_Moire,machine)
ind_TVB = n_cells*28    #top valence band
ind_LVB = n_cells*22    #lowest considered VB
if not Path(en_fn).is_file() or not Path(wg_fn).is_file():
    print("Computing en,wg...")
    energies = np.zeros((k_pts,ind_TVB-ind_LVB))
    weights = np.zeros((k_pts,ind_TVB-ind_LVB))
    look_up = fs.lu_table(N)
    for i in tqdm(range(k_pts)):
        K_i = K_list[i]
        H_tot = fs.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
        energies[i,:],evecs = eigh(H_tot,subset_by_index=[ind_LVB,ind_TVB-1])           #Diagonalize to get eigenvalues and eigenvectors
        for e in range(ind_TVB-ind_LVB):
            for l in range(2):
                for d in range(22):
                    weights[i,e] += np.abs(evecs[d+n_cells*l,e])**2
    if save:
        np.save(en_fn,energies)
        np.save(wg_fn,weights)
else:
    energies = np.load(en_fn)
    weights = np.load(wg_fn)

if 0: #plot some bands
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(ind_TVB-ind_LVB):
        plt.plot(K_list[:,0],energies[:,i])
    plt.ylim(Em,EM)
    plt.show()
    #exit()

#Compute spread and final picture
spread_k = 0.01
spread_E = 0.01
type_spread = 'Gauss'
pars_spread = (spread_k,spread_E,type_spread)
#
e_pts = exp_pic.shape[0]//pixel_factor
E_list = np.linspace(Em,EM,e_pts)
kkk = K_list[:,0]

spread_fn = fs.get_spread_fn(DFT,N,pars_V,pixel_factor,a_Moire,pars_spread,machine)
if not Path(spread_fn).is_file():
    print("Computing spreading...")
    spread = np.zeros((k_pts,e_pts))
    for i in tqdm(range(k_pts)):
        for n in range(ind_TVB-ind_LVB):
            spread += fs.weight_spreading(weights[i,n],K_list[i,0],energies[i,n],kkk[:,None],E_list[None,:],pars_spread)
    #Normalize in color scale
    norm_spread = fs.normalize_spread(spread,k_pts,e_pts)
    #en_cut = fs.normalize_cut(en_cut,pars_grid)
    if save:
        np.save(spread_fn,norm_spread)
else:
    norm_spread = np.load(spread_fn)

if 1:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18,10))
    plt.imshow(norm_spread,cmap='gray')
    plt.title(title)
    plt.xticks([0,norm_spread.shape[1]//2,norm_spread.shape[1]],["{:.2f}".format(-K),'0',"{:.2f}".format(K)])
    plt.yticks([0,norm_spread.shape[0]//2,norm_spread.shape[0]],["{:.2f}".format(EM),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(Em)])
    plt.xlabel("$A^{-1}$",size=15)
    plt.ylabel("$E\;(eV)$",size=15)
    if machine == 'loc':
        plt.show()
    else:
        plt.savefig(fs.get_fig_fn(DFT,N,pars_V,pixel_factor,a_Moire,pars_spread,machine))






