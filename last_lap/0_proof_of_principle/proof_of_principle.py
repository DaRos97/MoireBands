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
Extract KGK plot with different interlayers
"""
#Moire parameters
N = 4                               #####################
n_cells = int(1+3*N*(N+1))
#Model parameters
type_of_stacking = 'P' if int(sys.argv[1])<len(fs.list_f['P']) else 'AP'
m1,m2,mu = (0.13533,0.53226,-1.16385)
a,b,c = (-0.04,-0.03,-0.3) if type_of_stacking=='P' else (0,-0.08,-0.3)
V,phi = (0.02,1)

f1 = fs.list_f['P'][int(sys.argv[1])] if type_of_stacking=='P' else fs.list_f['P'][(int(sys.argv[1])-len(fs.list_f['P']))%len(fs.list_f['AP'])]
f2 = f1 if type_of_stacking == 'P' else f1*((int(sys.argv[1])-len(fs.list_f['P']))//len(fs.list_f['AP']))

all_pars = (type_of_stacking,m1,m2,mu,a,b,c,f1,f2,N,V,phi)
a_Moire = 79.8
G_M = fs.get_Moire(a_Moire)

title = "Type of stacking: "+type_of_stacking+'\n'
title += 'm1: '+"{:.3f}".format(m1)+', m2: '+"{:.3f}".format(m2)+r', $\mu$: '+"{:.3f}".format(mu)+'\n'
title += 'a: '+"{:.3f}".format(a)+', b: '+"{:.3f}".format(b)+', c: '+"{:.3f}".format(c)+'\n'
title += 'V: '+"{:.3f}".format(V)+r', $\phi$: '+"{:.3f}".format(phi)+'\n'
title += 'f1: '+"{:.3f}".format(f1)+', f2: '+"{:.3f}".format(f2)
print(title)
#exit()
#
if 0:
    #Extract S11 image
    S11_fn = fs.get_S11_fn(machine)
    K = 4/3*np.pi/fs.dic_params_a_mono['WSe2']
    EM = -0.5
    Em = -2.5
    bounds = (K,EM,Em)
    exp_pic = fs.extract_png(S11_fn,[-K,K,EM,Em])
    pixel_factor = 15            ###################################
    #BZ cut parameters
    cut = 'KGK'
    k_pts = exp_pic.shape[1]//pixel_factor
    K_list = fs.get_K(cut,k_pts)
    e_pts = exp_pic.shape[0]//pixel_factor
    E_list = np.linspace(Em,EM,e_pts)
    kkk = K_list[:,0]
else:
    EM = -0.8
    Em = -2
    K = 0.5
    k_pts = 200
    K_list = np.zeros((k_pts,2))
    K_list[:,0] = np.linspace(-K,K,k_pts)
    e_pts = 200
    E_list = np.linspace(Em,EM,e_pts)
    kkk = K_list[:,0]

#Compute energies and weights along KGK
en_fn = fs.get_energies_fn(all_pars,machine)
wg_fn = fs.get_weights_fn(all_pars,machine)
if not Path(en_fn).is_file() or not Path(wg_fn).is_file():
    print("Computing en,wg...")
    energies = np.zeros((k_pts,2*n_cells))
    weights = np.zeros((k_pts,2*n_cells))
    look_up = fs.lu_table(N)
    for i in tqdm(range(k_pts)):
        K_i = K_list[i]
        H_tot = fs.big_H(K_i,look_up,all_pars,G_M)
        energies[i,:],evecs = eigh(H_tot)           #Diagonalize to get eigenvalues and eigenvectors
        for e in range(2*n_cells):
            for l in range(2):
                weights[i,e] += np.abs(evecs[n_cells*l,e])**2
    if save:
        np.save(en_fn,energies)
        np.save(wg_fn,weights)
else:
    energies = np.load(en_fn)
    weights = np.load(wg_fn)

if 0: #plot some bands
    import matplotlib.pyplot as plt
    plt.figure()
    for e in range(2*n_cells):
        plt.plot(K_list[:,0],energies[:,e],linewidth=0.1,color='k')
        plt.scatter(K_list[:,0],energies[:,e],s=weights[:,e],color='b',marker='o')
    plt.xlim(-K,K)
    plt.ylim(Em,EM)
    plt.show()
    exit()

#Compute spread and final picture
spread_k = 0.005
spread_E = 0.005
type_spread = 'Gauss'
pars_spread = (spread_k,spread_E,type_spread)
#
spread_fn = fs.get_spread_fn(all_pars,pars_spread,machine)
if not Path(spread_fn).is_file():
    print("Computing spreading...")
    spread = np.zeros((k_pts,e_pts))
    for i in tqdm(range(k_pts)):
        for n in range(2*n_cells):
            spread += fs.weight_spreading(weights[i,n],K_list[i,0],energies[i,n],kkk[:,None],E_list[None,:],pars_spread)
    #Normalize in color scale
    norm_spread = fs.normalize_spread(spread,k_pts,e_pts)
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






