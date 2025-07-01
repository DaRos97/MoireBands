"""
Here we compute the final image with moiré replicas. There are a lot of parameters that enter this image.
For the monolayer parameters we take either the 'DFT' or the 'fit' values.
For the interlayer parameters we take the form of the interlayer interaction to have either 'C6' or 'C3' symmetry.
For the moiré potential we take different values of amplitue and phase at Gamma and K.
We consider different twist angles between the layers.
We can compare the result with sample S3 or S11 (this influences the choice of twist angle).
We take N circles of mini-BZs around the central one.
"""
import sys,os
import numpy as np
import scipy
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_moire as fsm
from pathlib import Path
import matplotlib.pyplot as plt
from time import time
from matplotlib.colors import Normalize
machine = cfs.get_machine(cwd)
if machine=='loc':
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm
disp = True

if len(sys.argv)!=2:
    print("Usage: py moire.py arg1")
    print("arg1: int -> index of parameters (see functions_moire.py)")
    exit()
#
plot_BZ_path = 0
save_en_wh = 1
plot_superimposed = 0
show_superimposed = 0
save_fig_superimposed = 0
save_spread = 1
plot_spread = 1
show_spread = 1
save_fig_spread = 1
save_spread_txt = 0
#
ind_pars = int(sys.argv[1])  #index of parameters
if machine == 'maf':
    ind_pars -= 1
monolayer_type, interlayer_symmetry, Vg, Vk, phiG, phiK, theta, sample, N, cut, k_pts, weight_exponent = fsm.get_pars(ind_pars)
#Monolayer parameters
pars_monolayer = fsm.import_monolayer_parameters(monolayer_type,machine)
#Interlayer parameters
pars_interlayer = [interlayer_symmetry,np.load(fsm.get_pars_interlayer_fn(sample,interlayer_symmetry,monolayer_type,machine))]
#Moire parameters
pars_moire = fsm.import_moire_parameters(N,(Vg,Vk,phiG,phiK),theta)
#Cut parameters
K_list = cfs.get_K(cut,k_pts)
#Spread image parameters
spread_k,spread_E,type_spread,deltaE,E_min,E_max = (
    1e-3,
    3e-2,   #5e-2,
    'Gauss',
    0.01,
    -3 if sample=='S11' else -2.5,
    -0.5 if sample=='S11' else 0
)
pars_spread = (spread_k,spread_E,type_spread,deltaE,E_min,E_max)
#Filenames
data_fn,spread_fn,spread_fig_fn = fsm.get_data_fns(fsm.get_pars(ind_pars),pars_spread,machine)

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
    if plot_BZ_path:
        fig = plt.figure()
        ax = fig.add_subplot()
        terms = cut.split('-')
        pt = k_pts//(len(terms)-1)
        for t in range(len(terms)-1):
            ax.scatter(K_list[pt*t:pt*(t+1),0],K_list[pt*t:pt*(t+1),1])
        ax.set_title(cut,size=20)
        ax.set_aspect('equal')
        plt.show()
        exit()

n_E = 5     #number of y-ticks in image
s_ = 20     #fontsize
xticks = []
xticks_labels = cut.split('-')
for i in range(len(cut.split('-'))):
    xticks.append(k_pts//(len(cut.split('-'))-1)*i)
###################################################################################
###################################################################################
if not Path(data_fn).is_file():
    energies = np.zeros((k_pts,pars_moire[1]*44))
    weights = np.zeros((k_pts,pars_moire[1]*44))
    look_up = fsm.lu_table(pars_moire[0])
    for i in tqdm(range(k_pts)):
        K_i = K_list[i]
        H_tot = fsm.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
        energies[i,:],evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
        ab = (np.absolute(evecs)**2)**weight_exponent
        ind_MB = 22 #index of main band of the layer
        weights[i,:] = np.sum(ab[:ind_MB,:],axis=0) + np.sum(ab[ind_MB*pars_moire[1]:ind_MB*pars_moire[1]+ind_MB,:],axis=0)
    if save_en_wh:
        en_wh = np.zeros((2,k_pts,pars_moire[1]*44))
        en_wh[0] = energies
        en_wh[1] = weights
        np.save(data_fn,en_wh)
else:
    en_wh = np.load(data_fn)
    energies = en_wh[0]
    weights = en_wh[1]
#
if plot_superimposed:   #Plot bands and weights superimposed to exp picture
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()
    if cut=='Kp-G-K' and 0:   #plot experimental image underneath
        exp_fn = 'inputs/S11_KGK_zoom.png' if sample=='S11' else 'inputs/S3_KGK.png'
        sname = 'S11zoom' if sample=='S11' else sample
        E_max,E_min = cfs.dic_energy_bounds[sname]
        bounds = [-np.linalg.norm(K_list[0]),np.linalg.norm(K_list[0]),E_max,E_min]
        pic = fsm.extract_png(exp_fn,bounds,sname)
        ax.imshow(pic,zorder=0)
        pe,pk,z = pic.shape
    else:
        pe,pk = (1000,k_pts)
    for e in range(10*pars_moire[1]):
        color = 'r'
        ax.plot(np.arange(k_pts)/k_pts*pk,
                (E_max-energies[:,18*pars_moire[1]+e])/(E_max-E_min)*pe,
                color=color,
                lw=0.05,
                zorder=2
                )
        color = 'b'
        ax.scatter(np.arange(k_pts)/k_pts*pk,
                (E_max-energies[:,18*pars_moire[1]+e])/(E_max-E_min)*pe,
                s=weights[:,18*pars_moire[1]+e]*50,#**(weight_exponent)*50,
                lw=0,
                color=color,
                zorder=3
                )
    ax.set_ylabel(r"$E\;(eV)$",size=20)
    ax.set_yticks(list(np.linspace(pe,0,n_E)),["{:.2f}".format(np.linspace(E_min,E_max,n_E)[i])for i in range(n_E)] )
    ax.set_ylim(pe,0)
    ax.set_xticks(xticks,xticks_labels,size=s_)
    ax.set_xlim(0,k_pts)

    fig.tight_layout()
    if save_fig_superimposed:
        figname = spread_fig_fn[:-4] + '_superimposed.png'
        plt.savefig(figname)
    if show_superimposed:
        plt.show()
    plt.close()
    if not plot_spread:
        exit()

##########################################################################
##########################################################################
##########################################################################

if disp:
    print("Computing spreading image with paramaters:")
    print("Spread function: ",type_spread)
    print("Spread in K: ","{:.5f}".format(spread_k)," 1/a")
    print("Spread in E: ","{:.5f}".format(spread_E)," eV")

E_list = np.linspace(E_min,E_max,int((E_max-E_min)/deltaE))
if not Path(spread_fn).is_file():
    print("Computing spreading...")
    spread = np.zeros((k_pts,len(E_list)))
    for i in tqdm(range(k_pts)):
        for n in range(pars_moire[1]*15,pars_moire[1]*28):
            spread += fsm.weight_spreading(weights[i,n],K_list[i],energies[i,n],K_list,E_list[None,:],pars_spread[:3])
    if save_spread:
        np.save(spread_fn,spread)
else:
    spread = np.load(spread_fn)

if plot_spread:
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()
    spread /= np.max(spread)        #0 to 1
    map_ = 'gray_r'
    #
    ax.imshow(spread.T[::-1,:]**0.5,
              cmap=map_,
              aspect=k_pts/len(E_list),
#              aspect='auto',
              interpolation='none'
             )
    #
    ax.set_xticks(xticks,xticks_labels,size=s_)
#    ax.set_xlim

    ax.set_ylabel("energy (eV)",size=s_)
    ax.set_yticks(list(np.linspace(len(E_list),0,n_E)),
                  ["{:.2f}".format(np.linspace(E_min,E_max,n_E)[i])for i in range(n_E)],
                 size=s_)
    fig.tight_layout()
    if save_fig_spread:
        fig.savefig(spread_fig_fn)
    if show_spread:
        plt.show()
    plt.close()

if save_spread_txt:
    print("Saving in txt format")
    fn = spread_fn[:-4]+'.txt'
    if not Path(fn).is_file():
        with open(fn,'w') as f:
            for k in range(k_pts):
                for e in range(len(E_list)):
                    f.write("{:.4f}".format(K_list[k,0])+','+"{:.4f}".format(K_list[k,1])+','+"{:.4f}".format(E_list[e])+","+"{:.7f}".format(spread[k,e])+'\n')




