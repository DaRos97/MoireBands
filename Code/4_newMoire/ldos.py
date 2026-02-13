""" LDOS in real space for selected parameters.
"""
import sys,os
import argparse
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
from tqdm import tqdm
machine = cfs.get_machine(os.getcwd())

""" Parameters and options """
parser = argparse.ArgumentParser(description="LDOS of final set of parameters")
parser.add_argument("Sample", help="Sample to consider (S3 or S11)")
parser.add_argument("Phase", help="Phase (deg°) solution to consider (60 for pi/3 or 180 for pi, check to have solutions for other angles)", type=float)
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()

sample = inputArguments.Sample
phiG = inputArguments.Phase/180*np.pi
disp = inputArguments.verbose

""" Fixed parameters """
nShells = 2
kPts = 30       # grid kPts x kPts in evals
rPts = 100#150
monolayer_type = 'fit'
Vk,phiK = (0.007,-106/180*np.pi)
nCells = int(1+3*nShells*(nShells+1))
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees

if inputArguments.Phase==60:
    w1p = -1.7 if sample=='S3' else -1.73
    w1d = 0.38 if sample=='S3' else 0.39
elif inputArguments.Phase==180:
    w1p = -1.71 if sample=='S3' else -1.73
    w1d = 0.38 if sample=='S3' else 0.39
elif inputArguments.Phase==175:
    w1p = -1.71 if sample=='S3' else -1.74
    w1d = 0.38 if sample=='S3' else 0.39
if sample=='S3' and phiG==np.pi:
    phiG = 175/180*np.pi
elif inputArguments.Phase==165:
    w1p = -1.73
    w1d = 0.39
    if sample=='S3':
        exit()

stacking = 'P'
w2p=w2d = 0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
G_M = cfs.get_reciprocal_moire(theta/180*np.pi)     #7 reciprocal moire lattice vectors
a_M = fsm.get_rs_moire(G_M)        #real space moirè vectors

if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Interlayer coupling w1: %f, %f"%(w1p,w1d))
    print("Sample ",sample," which has twist ",theta,"°")
    print("Moiré potential at K (%f eV, %f°)"%(Vk,phiK/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)
    print("(stacking,w2_p,w2_d,phi) = (%s, %.4f eV, %.4f eV, %.1f°)"%(stacking,w2p,w2d,phiG/np.pi*180))

""" Import best V from EDC fitting """
Vg = -1
data_fn = 'Data/EDC/Vbest_'+fsm.get_fn(*(sample,nShells,theta))+'.svg'
if Path(data_fn).is_file():
    with open(data_fn,'r') as f:
        l = f.readlines()
        for i in l:
            terms = i.split(',')
            if terms[0]==stacking and terms[1]=="{:.7f}".format(w1p) and terms[2]=="{:.7f}".format(w1d) and terms[3]=="{:.7f}".format(phiG):
                Vg = float(terms[-1])
                break
else:
    print("Data file not found: ",data_fn)
    quit()
if Vg==-1:
    print("Values of stacking,w1p,w1d and phiG not found in fit: %s, %.3f, %.3f, %.1f"%(stacking,w1p,w1d,phiG/np.pi*180))
    quit()

""" Evals and evecs """
kFlat = fsm.LDOS_kList(kPts,G_M)
args_e_data = (sample,nShells,monolayer_type,Vk,phiK,theta,stacking,w1p,w1d,phiG,kPts)
th_e_data_fn = 'Data/ldos_data_e_'+fsm.get_fn(*args_e_data)+'.npz'
if not Path(th_e_data_fn).is_file():
    moire_pars = (Vg,Vk,phiG,phiK)
    args = (nShells, nCells, kFlat, monolayer_type, parsInterlayer, theta, moire_pars, '', False, True)
    evals, evecs = fsm.diagonalize_matrix(*args)
    np.savez(th_e_data_fn,evals=evals,evecs=evecs)
else:
    evals = np.load(th_e_data_fn)['evals']
    evecs = np.load(th_e_data_fn)['evecs']

""" LDOS """
rList = fsm.LDOS_rList(rPts,a_M)
eMin, eMax, ePts = (-0.8,-0.65,200)
eList = np.linspace(eMin,eMax,ePts)           #CHECK THIS
def lorentzian(E, E0, eta=0.002):
    return eta / (np.pi * ((E - E0)**2 + eta**2))

args_l_data = args_e_data + (rPts,eMin,eMax,ePts)
th_l_data_fn = 'Data/ldos_data_l_'+fsm.get_fn(*args_l_data)+'.npy'

if not Path(th_l_data_fn).is_file():
    LDOS = np.zeros((rPts,ePts))
    lu = fsm.lu_table(nShells)
    Kbs = np.zeros((nCells,2))
    for i in range(nCells):
        Kbs[i] = G_M[1]*lu[i][0] + G_M[2]*lu[i][1]

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
    np.save(th_l_data_fn,LDOS)
else:
    LDOS = np.load(th_l_data_fn)

if 0:
    """ Plot 1 - my way """
    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(np.linalg.norm(rList,axis=1),eList)
    rL = np.linalg.norm(rList[-1])
    mesh = ax.pcolormesh(X,Y,LDOS.T,cmap='viridis')
    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )
    if 0:
        cbar.set_ticks(
            ticks=[np.min(LDOS),np.max(LDOS)],
            labels=['Low', 'High'],
            fontsize=20
        )

    ax.set_xticks([0,rL//3,2*rL//3,rL],[r"W/W",r"Se/W",r"W/S",r"W/W",],size=30)
    ax.set_ylabel('Energy [eV]',size=30)
    title = "LDOS sample %s"%sample[1:]
    ax.set_title(title,size=30)

    fig.tight_layout()
    plt.show()

    if input("Save?[y/N]")=='y':
        figname = "Figures/LDOS_"+fsm.get_fn(*args_l_data) + '.png'
        fig.savefig(figname)

if 1:
    """ Plot 2 - Li et al. """
    fig = plt.figure(figsize=(8,5))

    ax = fig.add_subplot(111)
    X,Y = np.meshgrid(eList,np.linalg.norm(rList,axis=1))
    rL = np.linalg.norm(rList[-1])
    mesh = ax.pcolormesh(X,Y,LDOS[::-1,:],cmap='hot')
    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    if 1:
        cbar.set_ticks(
            ticks=[np.min(LDOS),np.max(LDOS)],
            labels=['Low', 'High'],
            fontsize=20
        )

    ax.set_yticks([0,rL//3,2*rL//3,rL],[r"W/W",r"W/S",r"Se/W",r"W/W",],size=30)     #Changed order because we're plotting the y-axis inverted
    ax.set_xlabel('Energy [eV]',size=30)
    title = "LDOS sample %s"%sample[1:]
    ax.set_title(title,size=30)

    fig.tight_layout()
    plt.show()

    if input("Save?[y/N]")=='y':
        figname = "Figures/LDOS2_"+fsm.get_fn(*args_l_data) + '.png'
        fig.savefig(figname)










