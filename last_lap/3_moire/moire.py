"""
Here we compute the final image with moiré replicas. There are a lot of parameters that enter this image.
For the monolayer parameters we take either the 'DFT' or the 'fit values'.
For the interlayer parameters we take the form of the interlayer interaction to have either 'C6' or 'C3' symmetry.
For the moiré potential we take different values of amplitue and phase at gamma and K.
We consider different twist angles between the layers.
We can compare the result with sample S3 or S11 (this influences the choice of twist angle).
We take N circles of mini-BZs around the central one.
"""

import sys,os
import numpy as np
import scipy
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions3 as fs3
from pathlib import Path
import matplotlib.pyplot as plt
from time import time

machine = cfs.get_machine(cwd)
if machine=='loc':
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm
save_data = True
save_fig = False
disp = True
disp_plot = False
plot_superimposed = False
#
ind_pars = 0 if len(sys.argv)==1 else int(sys.argv[1])  #index of parameters
if machine == 'maf':
    ind_pars -= 1
monolayer_type, interlayer_symmetry, Vg, Vk, phiG, phiK, theta, sample, N, cut, k_pts = fs3.get_pars(ind_pars)
#
#Monolayer parameters
pars_monolayer = fs3.import_monolayer_parameters(monolayer_type,machine)
#Interlayer parameters
pars_interlayer = [interlayer_symmetry,np.load(fs3.get_pars_interlayer_fn(sample,interlayer_symmetry,monolayer_type,machine))]
#Moire parameters
pars_moire = fs3.import_moire_parameters(N,(Vg,Vk,phiG,phiK),theta)
#Cut parameters
K_list = cfs.get_K(cut,k_pts)
#Final image parameters
weight_exponent = 1/2       #exponent of weight -> should be 1 to be coherent
#Filenames
energies_fn,weights_fn,spread_fn = fs3.get_data_fns(fs3.get_pars(ind_pars),weight_exponent,machine)
#Compute spread and final picture
pars_spread = (0.01,0.03,'Gauss',0.01)   #spread_k,spread_E,type_spread,deltaE

spread_fn = fs3.get_spread_fn(DFT,N,pars_V,pixel_factor,a_Moire,txt_interlayer_symmetry[:2],pars_spread,weight_exponent,machine)

if disp:    #print what parameters we're using
    print("-----------PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Symmetry of interlayer coupling: ",interlayer_symmetry," with values from sample ",sample)
    print("Moiré potential values (eV,deg): G->("+"{:.4f}".format(Vg)+","+"{:.1f}".format(phiG/np.pi*180)+"°), K->("
          +"{:.4f}".format(Vk)+","+"{:.1f}".format(phiK/np.pi*180)+"°)")
    print("Twist angle: "+"{:.2f}".format(theta)+"° and moiré length: "+"{:.4f}".format(cfs.moire_length(theta/180*np.pi))+" A")
    print("Number of mini-BZs circles: ",N)
    print("Exponent of bands' weights: "+"{:.3f}".format(weight_exponent))
    print("Computing over BZ cut: ",cut," with ",k_pts," points")
    if disp_plot:
        fig = plt.figure()
        ax = fig.add_subplot()
        terms = cut.split('-')
        pt = k_pts//(len(terms)-1)
        for t in range(len(terms)-1):
            ax.scatter(K_list[pt*t:pt*(t+1),0],K_list[pt*t:pt*(t+1),1])
        ax.set_title(cut,size=20)
        plt.show()
###################################################################################
###################################################################################
if not Path(energies_fn).is_file():
    energies = np.zeros((k_pts,pars_moire[1]*44))
    weights = np.zeros((k_pts,pars_moire[1]*44))
    look_up = fs3.lu_table(pars_moire[0])
    for i in tqdm(range(k_pts)):
        K_i = K_list[i]
        H_tot = fs3.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
        energies[i,:],evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
        ab = np.absolute(evecs)**2
        weights[i,:] = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*pars_moire[1]:22*pars_moire[1]+22,:],axis=0)
    if save_data:
        np.save(energies_fn,energies)
        np.save(weights_fn,weights)
else:
    energies = np.load(energies_fn)
    weights = np.load(weights_fn)
#
if plot_superimposed:   #Plot bands and weights superimposed to exp picture
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot()
    for e in range(6*pars_moire[1]):
        color = 'r'
        ax.plot(np.arange(k_pts),
                energies[:,22*pars_moire[1]+e],
                color=color,
                lw=0.05,
                zorder=2
                )
        color = 'b'
        ax.scatter(np.arange(k_pts),
                energies[:,22*pars_moire[1]+e],
                s=weights[:,22*pars_moire[1]+e]**(weight_exponent)*100,
                lw=0,
                color=color,
                zorder=3
                )
    ax.set_ylabel("$E\;(eV)$",size=20)
    fig.tight_layout()
    if save_fig:
        plt.savefig('results/figures/moire_twisted/'+title_fig+'.png')
    plt.show()

##########################################################################
##########################################################################
##########################################################################
if 0:
    exit()

E_list = np.linspace(-3,-0.5,e_pts)
kkk = K_list[:,0]
if not Path(spread_fn).is_file():
    print("Computing spreading...")
    spread = np.zeros((k_pts,e_pts))
    for i in tqdm(range(k_pts)):
        for n in range(ind_TVB-ind_LVB):
            spread += fs3.weight_spreading(weights[i,n]**(weight_exponent),K_list[i,0],energies[i,n],kkk[:,None],E_list[None,:],pars_spread)
    #Normalize in color scale
    norm_spread = fs3.normalize_spread(spread,k_pts,e_pts)
    if 1:
        np.save(spread_fn,norm_spread)
else:
    norm_spread = np.load(spread_fn)

fig,ax = plt.subplots(figsize=(14,9))
#cmaps: gray, viridis,
norm_spread /= np.max(norm_spread)

map_ = 'gray' #if len(sys.argv)<4 else sys.argv[3]
ax.imshow(norm_spread,cmap=map_)
ax.set_xticks([0,norm_spread.shape[1]//2,norm_spread.shape[1]],[r"$K'$",r'$\Gamma$',r'$K$'],size=25)
#ax.set_yticks([0,norm_spread.shape[0]//2,norm_spread.shape[0]],["{:.2f}".format(Em),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(EM)],size=25)
ax.set_yticks([])
ax.set_ylabel("Energy",size=25)
#ax.plot([0,0],[-10,10],color='r',lw=0.5,zorder=-1)
#ax.set_title(title_fig)
fig.tight_layout()
if save_fig:
    plt.savefig('results/figures/spread/'+title_fig+'.png')
if 1:#machine == 'loc':
    plt.show()






