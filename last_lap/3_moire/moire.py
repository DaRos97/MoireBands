import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions as fs
from pathlib import Path
from scipy.linalg import eigh
import matplotlib.pyplot as plt

machine = cfs.get_machine(os.getcwd())
if machine=='loc':
    from tqdm import tqdm
    save = False
else:
    tqdm = cfs.tqdm
    save = True
"""
Here we compute the full KGK image with Moire replicas.
"""

#Moire parameters
ind = 0 if len(sys.argv)==1 else int(sys.argv[1])
if machine == 'maf':
    ind -= 1
DFT = True
sample, interlayer_type, Vg, Vk, phiG, phiK = fs.get_pars(ind)
N = 1 if len(sys.argv)<3 else int(sys.argv[2])
pixel_factor = 5
n_cells = int(1+3*N*(N+1))
zoom = True if sample=='S11' else False
"""
Moirè potential of bilayer
Different at Gamma (d_z^2 orbital) -> first two parameters, and K (d_xy orbitals) -> last two parameters
Gamma point values from paper "G valley TMD moirè bands" (first in eV, second in radiants)
K point values from Louk's paper (first in eV, second in radiants)
"""
pars_V = (Vg,Vk,phiG,phiK)
t_twist = cfs.dic_params_twist[sample][1]*np.pi/180
a_Moire = cfs.moire_length(t_twist)
#
txt_dft = 'DFT' if DFT else 'fit'
title = "sample_"+sample+",N_"+str(N)+",tb_pars_"+txt_dft+',interlayer_'+interlayer_type+",_pars_V_"+fs.get_list_fn(pars_V)+",a_Moire_"+"{:.4f}".format(a_Moire)
print(title)
#
"""
Hamiltonian of Moire interlayer (diagonal with correct signs of phase)
Compute it here because is k-independent.
"""
G_M = fs.get_reciprocal_moire(t_twist)
H_moire = [fs.H_moire(0,pars_V),fs.H_moire(1,pars_V)]
pars_moire = (N,pars_V,G_M,H_moire)
#Monolayer parameters
hopping = {}
epsilon = {}
HSO = {}
offset = {}
for TMD in cfs.TMDs:
    temp = np.load(fs.get_pars_mono_fn(TMD,machine,DFT))
    hopping[TMD] = cfs.find_t(temp)
    epsilon[TMD] = cfs.find_e(temp)
    HSO[TMD] = cfs.find_HSO(temp[-2:])
    offset[TMD] = temp[-3]
pars_monolayer = (hopping,epsilon,HSO,offset)
#Interlayer parameters
pars_interlayer = [interlayer_type,np.load(fs.get_pars_interlayer_fn(sample,interlayer_type,DFT,machine))]

K0 = 4/3*np.pi/cfs.dic_params_a_mono['WSe2']
sample_fn = fs.get_sample_fn(sample,machine,zoom)
energy_bounds = {'S11': (-0.5,-2.5), 'S3': (-0.2,-1.8), 'S11zoom':(-0.7,-1.8)}
label = sample+'zoom' if zoom else sample
EM, Em = energy_bounds[label]
exp_pic = fs.extract_png(sample_fn,[-K0,K0,EM,Em],label)

#BZ cut parameters
cut = 'KGK'
k_pts = exp_pic.shape[1]//pixel_factor
K_list = fs.get_K(cut,k_pts)

if 1:# and machine=='loc':    #Compute (no-)moire image superimposed to experiment
    ens_temp_fn = 'results/E_data/E_'+title+'.npy'
    wei_temp_fn = 'results/E_data/W_'+title+'.npy'
    ind_TVB = n_cells*30    #top valence band
    ind_LVB = n_cells*22    #lowest considered VB
    if not Path(ens_temp_fn).is_file():
        energies = np.zeros((k_pts,ind_TVB-ind_LVB))
        weights = np.zeros((k_pts,ind_TVB-ind_LVB))
        look_up = fs.lu_table(N)
        pars_moire = (N,pars_V,G_M,H_moire)
        for i in tqdm(range(k_pts)):
            K_i = K_list[i]
            H_tot = fs.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
            energies[i,:],evecs = eigh(H_tot,subset_by_index=[ind_LVB,ind_TVB-1])           #Diagonalize to get eigenvalues and eigenvectors
            ab = np.absolute(evecs)**2
            weights[i,:] = np.sum(ab[:22,:ind_TVB-ind_LVB],axis=0) + np.sum(ab[22*n_cells:22*n_cells+22,:ind_TVB-ind_LVB],axis=0)
        np.save(ens_temp_fn,energies)
        np.save(wei_temp_fn,weights)
    else:
        energies = np.load(ens_temp_fn)
        weights = np.load(wei_temp_fn)
    #
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot()
    pe,pk,z = exp_pic.shape
    for e in range(ind_TVB-ind_LVB):
        color = 'r'
        ax.plot((K_list[:,0]+K0)/2/K0*pk,
                (EM-energies[:,e])/(EM-Em)*pe,
                color=color,
                lw=0.05,
                zorder=2
                )
        color = 'b'
        ax.scatter((K_list[:,0]+K0)/2/K0*pk,
                (EM-energies[:,e])/(EM-Em)*pe,
                s=weights[:,e]*100,
                lw=0,
                color=color,
                zorder=3
                )
    ax.imshow(exp_pic,zorder=1)
    ax.set_title(title)
    if machine=='loc':
        plt.show()
    plt.savefig('results/figures/moire_twisted/'+title+'.png')
    exit()

#Compute energies and weights along KGK
en_fn = fs.get_energies_fn(DFT,N,pars_V,pixel_factor,a_Moire,interlayer_type,machine)
wg_fn = fs.get_weights_fn(DFT,N,pars_V,pixel_factor,a_Moire,interlayer_type,machine)
ind_TVB = n_cells*28    #top valence band
ind_LVB = n_cells*24    #lowest considered VB
if not Path(en_fn).is_file() or not Path(wg_fn).is_file():
    print("Computing en,wg...")
    energies = np.zeros((k_pts,ind_TVB-ind_LVB))
    weights = np.zeros((k_pts,ind_TVB-ind_LVB))
    look_up = fs.lu_table(N)
    for i in tqdm(range(k_pts)):
        K_i = K_list[i]
        H_tot = fs.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
        energies[i,:],evecs = eigh(H_tot,subset_by_index=[ind_LVB,ind_TVB-1])           #Diagonalize to get eigenvalues and eigenvectors
        ab = np.absolute(evecs)**2
        weights[i,:] = np.sum(ab[:22,:ind_TVB-ind_LVB],axis=0) + np.sum(ab[22*n_cells:22*n_cells+22,:ind_TVB-ind_LVB],axis=0)
    if save:
        np.save(en_fn,energies)
        np.save(wg_fn,weights)
else:
    energies = np.load(en_fn)
    weights = np.load(wg_fn)

#Compute image of band weigths superimposed to experiment
plt.figure(figsize=(20,15))
px,py,z = exp_pic.shape
x_line = (K_list[:,0]-K_list[0,0])/(K_list[-1,0]-K_list[0,0])*py
for e in range(ind_TVB-ind_LVB):
    e_line = (energies[:,e]-Em)/(EM-Em)*px
#        plt.plot(x_line,e_line,color='r',zorder=1,linewidth=0.1)
    plt.scatter(x_line,e_line,s=weights[:,e]*200,lw=0,color='r',marker='o',zorder=3)
plt.ylim(0,px)
plt.imshow(exp_pic[::-1,:,:],zorder=-1)
plt.xticks([0,exp_pic.shape[1]//2,exp_pic.shape[1]],[r"$K'$",r'$\Gamma$',r'$K$'],size=20)
plt.yticks([0,exp_pic.shape[0]//2,exp_pic.shape[0]],["{:.2f}".format(Em),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(EM)])
plt.ylabel("$E\;(eV)$",size=15)
if machine == 'loc':
    plt.show()
else:
    fig1_fn = fs.get_fig1_fn(DFT,N,pars_V,pixel_factor,a_Moire,interlayer_type,machine)
    plt.savefig(fig1_fn)
    plt.close()

#Compute spread and final picture
spread_k = 0.01
spread_E = 0.01
type_spread = 'Gauss'
pars_spread = (spread_k,spread_E,type_spread)
#
e_pts = exp_pic.shape[0]//pixel_factor
E_list = np.linspace(Em,EM,e_pts)
kkk = K_list[:,0]

spread_fn = fs.get_spread_fn(DFT,N,pars_V,pixel_factor,a_Moire,interlayer_type,pars_spread,machine)
if not Path(spread_fn).is_file():
    print("Computing spreading...")
    spread = np.zeros((k_pts,e_pts))
    for i in tqdm(range(k_pts)):
        for n in range(ind_TVB-ind_LVB):
            spread += fs.weight_spreading(weights[i,n],K_list[i,0],energies[i,n],kkk[:,None],E_list[None,:],pars_spread)
    #Normalize in color scale
    norm_spread = fs.normalize_spread(spread,k_pts,e_pts)
    if save:
        np.save(spread_fn,norm_spread)
else:
    norm_spread = np.load(spread_fn)

#Figure
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(14,9))
#cmaps: gray, viridis,
norm_spread /= np.max(norm_spread)

map_ = 'gray' if len(sys.argv)<4 else sys.argv[3]
ax.imshow(norm_spread,cmap=map_)
ax.set_xticks([0,norm_spread.shape[1]//2,norm_spread.shape[1]],[r"$K'$",r'$\Gamma$',r'$K$'],size=20)
ax.set_yticks([0,norm_spread.shape[0]//2,norm_spread.shape[0]],["{:.2f}".format(Em),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(EM)])
ax.set_ylabel("$E\;(eV)$",size=15)
ax.set_title(title)
if machine == 'loc':
    plt.show()
else:
    plt.savefig(fs.get_fig_fn(DFT,N,pars_V,pixel_factor,a_Moire,interlayer_type,pars_spread,machine))






