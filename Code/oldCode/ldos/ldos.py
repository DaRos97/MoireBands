"""
Here we compute the LDOS in real space and maybe even the chern number to construct the topological phase diagram.
For the LDOS the steps are the same as for the test on the pan paper.
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
sys.path.insert(1, master_folder+'/3_moire')
import CORE_functions as cfs
import functions_ldos as fsl
import functions_moire as fsm
from pathlib import Path
import matplotlib.pyplot as plt
machine = cfs.get_machine(cwd)
if machine=='loc':
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm

if len(sys.argv) not in [1,2]:
    print("Usage: python3 ldos.py arg1(optional=0)")
    print("\targ1: index of parameters")
    exit()
else:
    if len(sys.argv)==2:
        ind_pars = int(sys.argv[1])  #index of parameters
        if machine == 'maf':
            ind_pars -= 1
    else:
        ind_pars = 0

save_data = True
plot_ldos = True
save_plot_ldos = 1#False
show_plot_ldos = True


monolayer_type, interlayer_symmetry, Vg, Vk, phiG, phiK, theta, sample, nShells, kPts, rPts = fsl.get_pars(ind_pars)
nCells = int(1+3*nShells*(nShells+1))
#Monolayer parameters
pars_monolayer = fsm.import_monolayer_parameters(monolayer_type,machine)
#Interlayer parameters
pars_interlayer = [interlayer_symmetry,np.load(fsm.get_pars_interlayer_fn(sample,interlayer_symmetry,monolayer_type,machine))]
#Moire parameters
pars_moire = fsm.import_moire_parameters(nShells,(Vg,Vk,phiG,phiK),theta)
nShells,nCells,pars_V,G_M,H_moires = pars_moire
a_M = fsl.get_moire(G_M)        #real space moirè vectors

args_fn = (monolayer_type, interlayer_symmetry, Vg, Vk, phiG, phiK, theta, sample, nShells, kPts)
data_fn = "Data/evals_evecs_" + fsl.get_fn(*args_fn) + ".npz"

kFlat = fsl.compute_kList(kPts,G_M)
lu = fsl.lu_table(nShells)
if not Path(data_fn).is_file():
    evals = np.zeros((kPts**2,nCells*44))
    evecs = np.zeros((kPts**2,nCells*44,nCells*44),dtype=complex)
    for i in tqdm(range(kPts**2)):
        K_i = kFlat[i]
        H_tot = fsm.big_H(K_i,lu,pars_monolayer,pars_interlayer,pars_moire)
        evals[i],evecs[i] = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
    if save_data:
        np.savez(data_fn,evals=evals,evecs=evecs)
else:
    print("Already computed")
    evals = np.load(data_fn)['evals']
    evecs = np.load(data_fn)['evecs']

rList = fsl.compute_rList(rPts,a_M)

eList = np.linspace(-1,0,200)

def lorentzian(E, E0, eta=0.002):
    return eta / (np.pi * ((E - E0)**2 + eta**2))

LDOS = np.zeros((rPts,len(eList)))
Gs = [np.zeros(2),]
for i in range(6):
    Gs.append(cfs.R_z(np.pi/3*i)@G_M[0])

Kbs = np.zeros((nCells,2))
for i in range(nCells):
    Kbs[i] = Gs[1]*lu[i][0] + Gs[2]*lu[i][1]

ig = np.arange(nCells)[np.newaxis, :]  # (1, nCells)
alpha = np.arange(44)[:, np.newaxis]  # (44, 1)       #index over alphas -> orbitals
ind = (alpha % 22) + ig * 22 + nCells * 22 * (alpha // 22)  #44,nCells
for ik in tqdm(range(kPts**2), desc="Momentum iteration"):
    evals_k = evals[ik]
    evecs_k = evecs[ik]
    kGs = Kbs + kFlat[ik]        #nCells,2
    phases = np.exp(1j * rList @ kGs.T)[np.newaxis,:,:]     #1,nR,nCells
    for n,En in enumerate(evals_k):
        #Vectorized calculation
        coeffs = evecs_k[ind,n]
        coeffs_all = coeffs[:,np.newaxis,:]  # (44, 1, nCells)
        #
        psi_alpha = np.sum(phases * coeffs_all, axis=-1)  # (44, nR)
        psi_r_all = np.sum(np.abs(psi_alpha)**2, axis=0)  # nR
        lorentz_matrix = lorentzian(eList, En)  # nE
        LDOS += psi_r_all[:,None] * lorentz_matrix[None,:]  # (nR, nE)


LDOS /= kPts**2

if plot_ldos:
    fig = plt.figure(figsize=(20,20))
#    ax = fig.add_subplot(121)
#    ax.plot()

    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(np.linalg.norm(rList,axis=1),eList)
    mesh = ax.pcolormesh(X,Y,LDOS.T,cmap='viridis')
#    fig.colorbar(mesh,ax=ax)

    ax.set_xlabel('position')
    ax.set_ylabel('energy')
    title = "V_g="+"{:.5f}".format(Vg)+"_phase="+"{:.3f}".format(phiG)
    ax.set_title(title)

    if save_plot_ldos:
        figure_fn = "Figures/LDOS_me_"+fsl.get_fn(*args_fn)+".png"
        plt.savefig(figure_fn)
    if show_plot_ldos:
        plt.show()
    plt.close()






































