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
import functions3 as fs
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
ind_pars = 0 if len(sys.argv)==1 else int(sys.argv[1])  #index of parameters
if machine == 'maf':
    ind_pars -= 1
DFT, C3, sample, Vg, Vk, phiG, phiK, ind_theta = fs.get_pars(ind_pars)
interlayer_type = 'C3' if sample=='S3' else 'C6'
N = 1 if len(sys.argv)<3 else int(sys.argv[2])      #number of circles of mBZ
n_cells = int(1+3*N*(N+1))
zoom = True if sample=='S11' else False     #Use zoomed image of S11 which has higher contrast
label = sample+'zoom' if zoom else sample
"""
Moirè potential of bilayer
Different at Gamma (d_z^2 orbital) and K (d_xy orbitals)
Gamma point values from paper: M.Angeli et al., Proceedings of the National Academy of Sciences 118.10 (2021): e2021826118.
K point values from Louk's paper: L.Rademaker, Phys. Rev. B 105, 195428 (2022)
"""
pars_V = (Vg,Vk,phiG,phiK)
t_twist = np.linspace(cfs.dic_params_twist[sample][0],cfs.dic_params_twist[sample][-1],3)[ind_theta]*np.pi/180#cfs.dic_params_twist[sample][ind_theta]*np.pi/180     #use best estimate of twist angle, depending on the sample
a_Moire = cfs.moire_length(t_twist)
weight_exponent = 1/2
#C3 = False
txt_C3 = 'C3_symm' if C3 else 'C6_symm'
#
txt_dft = 'DFT' if DFT else 'fit'
title_E = "sample_"+sample+",N_"+str(N)+",tb_pars_"+txt_dft+',interlayer_'+interlayer_type+",_pars_V_"+fs.get_list_fn(pars_V)+",twist_"+"{:.4f}".format(t_twist*180/np.pi)+'_'+txt_C3
title_fig = title_E +"_weight_"+"{:.3f}".format(weight_exponent)
print(title_fig)
#
"""
Hamiltonian of Moire interlayer (diagonal with correct signs of phase)
Compute it here because is k-independent.
"""
G_M = fs.get_reciprocal_moire(t_twist)
H_moire = [fs.H_moire(0,pars_V),fs.H_moire(1,pars_V)]
pars_moire = (N,pars_V,G_M,H_moire,C3)
#Monolayer parameters
hopping = {}
epsilon = {}
HSO = {}
offset = {}
for TMD in cfs.TMDs:
    DFT_1 = DFT if TMD=='WSe2' else True    #Use DFT for WS2 in order to avoid the conduction band to touch the valence band of WSe2
    temp = np.load(fs.get_pars_mono_fn(TMD,machine,DFT_1))
    if not DFT_1:
        temp = np.append(temp,np.load(fs.get_SOC_fn(TMD,machine)))
    hopping[TMD] = cfs.find_t(temp)
    epsilon[TMD] = cfs.find_e(temp)
    HSO[TMD] = cfs.find_HSO(temp[-2:])
    offset[TMD] = temp[-3]
pars_monolayer = (hopping,epsilon,HSO,offset)
#Interlayer parameters
pars_interlayer = [interlayer_type,np.load(fs.get_pars_interlayer_fn(sample,interlayer_type,DFT,machine))]
#Image properties
K0 = 4/3*np.pi/cfs.dic_params_a_mono['WSe2']
sample_fn = fs.get_sample_fn(sample,machine,zoom)
energy_bounds = {'S11': (-0.5,-2.5), 'S3': (-0.2,-1.8), 'S11zoom':(-0.7,-1.8)}
EM, Em = energy_bounds[label]
exp_pic = fs.extract_png(sample_fn,[-K0,K0,EM,Em],label)

#BZ cut parameters
cut = 'KGK'
k_pts = 400#exp_pic.shape[1]//pixel_factor
K_list = fs.get_K(cut,k_pts)

#Compute energy and weights
ens_temp_fn = 'results/E_data/E_'+title_E+'.npy'
wei_temp_fn = 'results/E_data/W_'+title_E+'.npy'
ind_TVB = n_cells*28    #top valence band
ind_LVB = n_cells*24    #lowest considered VB
if not Path(ens_temp_fn).is_file():
    energies = np.zeros((k_pts,ind_TVB-ind_LVB))
    weights = np.zeros((k_pts,ind_TVB-ind_LVB))
    look_up = fs.lu_table(N)
    pars_moire = (N,pars_V,G_M,H_moire,C3)
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
#Relative weight of side band
if 1:
    pass
#Plot bands and weights superimposed to exp picture
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
            s=weights[:,e]**(weight_exponent)*100,
            lw=0,
            color=color,
            zorder=3
            )
if 0:
    color = 'r'
    nks = [145,160]
    es = [ind_TVB-ind_LVB-4+np.argmax(weights[nks[0],-4:-2]),np.argmax(weights[nks[1],:])]
    for i in range(2):
        e = es[i]
        nk = nks[i]
    #    print(weights[nk,:])
        ax.scatter((K_list[nk,0]+K0)/2/K0*pk,
                   (EM-energies[nk,e])/(EM-Em)*pe,
                   s=weights[nk,e]**(weight_exponent)*100,
                lw=0,
                color=color,
                zorder=4
                )
    print("relative_weight: ",(np.max(weights[nks[0],-4:-2])/np.max(weights[nks[1],:]))**weight_exponent)
ax.imshow(exp_pic,zorder=1)
ax.set_xticks([0,exp_pic.shape[1]//2,exp_pic.shape[1]],[r"$K'$",r'$\Gamma$',r'$K$'],size=20)
ax.set_yticks([0,exp_pic.shape[0]//2,exp_pic.shape[0]],["{:.2f}".format(EM),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(Em)])
ax.set_ylabel("$E\;(eV)$",size=20)
ax.set_ylim(exp_pic.shape[0],0)
ax.set_title(title_fig,size=25)
fig.tight_layout()
plt.savefig('results/figures/moire_twisted/'+title_fig+'.png')
if 0:#machine=='loc':
    plt.show()
plt.close()

##########################################################################
##########################################################################
##########################################################################
if 1:
    exit()

#Compute spread and final picture
spread_k = 0.01
spread_E = 0.03
type_spread = 'Gauss'
pars_spread = (spread_k,spread_E,type_spread)
#
pixel_factor = 5
e_pts = exp_pic.shape[0]//pixel_factor
E_list = np.linspace(Em,EM,e_pts)
kkk = K_list[:,0]

spread_fn = fs.get_spread_fn(DFT,N,pars_V,pixel_factor,a_Moire,interlayer_type,pars_spread,weight_exponent,machine)
if not Path(spread_fn).is_file():
    print("Computing spreading...")
    spread = np.zeros((k_pts,e_pts))
    for i in tqdm(range(k_pts)):
        for n in range(ind_TVB-ind_LVB):
            spread += fs.weight_spreading(weights[i,n]**(weight_exponent),K_list[i,0],energies[i,n],kkk[:,None],E_list[None,:],pars_spread)
    #Normalize in color scale
    norm_spread = fs.normalize_spread(spread,k_pts,e_pts)
    if 1:
        np.save(spread_fn,norm_spread)
else:
    norm_spread = np.load(spread_fn)

fig,ax = plt.subplots(figsize=(14,9))
#cmaps: gray, viridis,
norm_spread /= np.max(norm_spread)

map_ = 'gray' #if len(sys.argv)<4 else sys.argv[3]
ax.imshow(norm_spread,cmap=map_)
ax.set_xticks([0,norm_spread.shape[1]//2,norm_spread.shape[1]],[r"$K'$",r'$\Gamma$',r'$K$'],size=20)
ax.set_yticks([0,norm_spread.shape[0]//2,norm_spread.shape[0]],["{:.2f}".format(Em),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(EM)])
ax.set_ylabel("$E\;(eV)$",size=15)
ax.set_title(title_fig)
if 1:#machine == 'loc':
    plt.show()
else:
    plt.savefig(fs.get_fig_fn(DFT,N,pars_V,a_Moire,interlayer_type,pars_spread,machine))






