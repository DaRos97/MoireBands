"""
Here we compute the LDOS in real space as the Pan paper, to test if I'm doing the right calculation.
"""
import sys,os
import numpy as np
import scipy
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions_ldos as fsl

save_data = True#False
plot_ldos = True
save_plot_ldos = False
show_plot_ldos = True

V = 0.005   #eV
psi = 0.5*np.pi
Vm = V*np.exp(-1j*psi)
theta = 3/180*np.pi
w = 0.02    #eV
kPts = 6
kPts2 = 500
l1 = int(kPts2/(3+np.sqrt(3)))   #length of k'+ -> g (same as  g -> k- and k- -> k+)
l2 = int(l1*np.sqrt(3))
kPts2 = 3*l1+l2
rPts = 100
a_0 = 3.28  #A
a_M = a_0/2/np.sin(theta/2)

en_coeff = scipy.constants.hbar**2/2/0.45/scipy.constants.m_e / scipy.constants.e * 1e20 #eV*A^2

nShells = 3
nCells = int(1+3*nShells*(nShells+1))

bs = [ np.zeros(2), np.array([1,0])*4*np.pi/np.sqrt(3)/a_M, ]
for i in range(1,6):
    bs.append( fsl.R_z(np.pi/3*i)@bs[1] )

args_fn = ('pan',V,psi,theta,w,kPts,kPts2,nShells)
data_fn = "Data/evals_evecs_"+fsl.get_fn(*args_fn)+".npz"

kFlat = fsl.compute_kList(kPts,bs[1:3])         #for sum over mini-BZ
kBands = fsl.compute_kBands(l1,l2,bs[1:3])      #for plotting the bands
lu = fsl.lu_table(nShells)
if not Path(data_fn).is_file():
    evals = np.zeros((kPts**2,2*nCells))
    evals_b = np.zeros((kPts2,2*nCells))
    evecs = np.zeros((kPts**2,2*nCells,2*nCells),dtype=complex)
    args = (lu,nShells,nCells,en_coeff,bs,Vm,w)
    for i in tqdm(range(kPts**2),desc="Momentum mesh diagonalization"):
        K_i = kFlat[i]
        H_tot = fsl.H_pan(K_i,*args)
        evals[i],evecs[i] = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
    for i in range(kPts2):
        K_i = kBands[i]
        H_tot = fsl.H_pan(K_i,*args)
        evals_b[i] = scipy.linalg.eigvalsh(H_tot)           #Diagonalize to get eigenvalues and eigenvectors
    if save_data:
        np.savez(data_fn,evals=evals,evecs=evecs,evals_b=evals_b)
else:
    print("Already computed")
    evals = np.load(data_fn)['evals']
    evals_b = np.load(data_fn)['evals_b']
    evecs = np.load(data_fn)['evecs']

if 0:#Lattice
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(1,7):
        ax.plot([0,bs[i][0]],[0,bs[i][1]],lw=2,label=str(i))
    ax.scatter(kBands[:kPts2//2,0],kBands[:kPts2//2,1],color='r')
    ax.scatter(kBands[kPts2//2:,0],kBands[kPts2//2:,1],color='b')
    ax.legend()
    ax.set_aspect('equal')
    plt.show()
    exit()

#LDOS

rList = fsl.compute_rList(rPts,fsl.get_moire(bs[1:3]))

eList = np.linspace(-0.02,0.04,200)

def lorentzian(E, E0, eta=0.002):
    return eta / (np.pi * ((E - E0)**2 + eta**2))

LDOS = np.zeros((rPts,len(eList)))

Kbs = np.zeros((nCells,2))
for i in range(nCells):
    Kbs[i] = bs[1]*lu[i][0] + bs[2]*lu[i][1]

ig = np.arange(nCells)[np.newaxis, :]  # (1, nCells)
alpha = np.arange(2)[:, np.newaxis]  # (2, 1)       #index over alphas -> orbitals
ind = ig + nCells * alpha  #2,nCells
for ik in tqdm(range(kPts**2), desc="Momentum mesh LDOS"):
    evals_k = evals[ik]
    evecs_k = evecs[ik]
    kGs = Kbs + kFlat[ik]        #nCells,2
    phases = np.exp(1j * rList @ kGs.T)[np.newaxis,:,:]     #1,nR,nCells
    for n,En in enumerate(evals_k):
        #Vectorized calculation
        coeffs = evecs_k[ind,n]
        coeffs_all = coeffs[:,np.newaxis,:]  # (2, 1, nCells)
        #
        psi_alpha = np.sum(phases * coeffs_all, axis=-1)  # (2, nR)
        psi_r_all = np.sum(np.abs(psi_alpha)**2, axis=0)  # nR
        lorentz_matrix = lorentzian(eList, En)  # nE
        LDOS += psi_r_all[:,None] * lorentz_matrix[None,:]  # (nR, nE)


LDOS /= kPts**2

if plot_ldos:
    fig = plt.figure(figsize=(10,20))
    ax = fig.add_subplot(211)
    kMod = np.arange(kPts2)
    for n in range(2*nCells-12,2*nCells):
        ax.plot(kMod,evals_b[:,n],color='b',lw=4)
    ax.axvline(l1,color='k')
    ax.axvline(2*l1,color='k')
    ax.axvline(3*l1,color='k')
    ax.set_ylim(-0.12,0.05)
    ax.set_xlim(0,kPts2-1)
    ax.set_xticks([0,l1,2*l1,3*l1,3*l1+l2-1],[r"$k'^+$",r'$\gamma$',r'$k^-$',r'$k^+$',r"$k'^+$"],size=15)
    ax.set_ylabel("Energy(eV)",size=15)
    #LDOS
    ax = fig.add_subplot(212)
    rMod = np.linalg.norm(rList,axis=1)
    X,Y = np.meshgrid(rMod,eList)
    mesh = ax.pcolormesh(X,Y,LDOS.T,cmap='viridis')
#    fig.colorbar(mesh,ax=ax)

    ax.set_xticks([0,rMod[rPts//3],rMod[rPts//3*2],rMod[rPts-1]],[r"$\mathcal{R}^M_M$",r"$\mathcal{R}^X_M$",r"$\mathcal{R}^M_X$",r"$\mathcal{R}^M_M$"],size=15)
    ax.set_ylabel("Energy(eV)",size=15)

    fig.tight_layout()
    if save_plot_ldos:
        figure_fn = "Figures/LDOS_pan_"+fsl.get_fn(*args_fn)+".png"
        plt.savefig(figure_fn)
    if show_plot_ldos:
        plt.show()
    plt.close()





















