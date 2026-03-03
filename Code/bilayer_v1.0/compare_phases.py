"""
Here we plot the same parameters with different moire phases to see how the spectral weight changes.
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
machine = cfs.get_machine(os.getcwd())

sample = 'S3'
nShells = 2
monolayer_type = 'fit'
Vk,phiK = (0.007,-106/180*np.pi)
nCells = int(1+3*nShells*(nShells+1))
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees
w1p = -1.7 if sample=='S3' else -1.73
w1d = 0.38 if sample=='S3' else 0.39
w2p = w2d = 0
stacking = 'P'
Vg = 0.016
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}

kPts = 101 #has to be odd
range_k = 0.5
kList = np.zeros((kPts,2))
kList[:,0] = np.linspace(-range_k,range_k,kPts)
kLine = kList[:,0]

listPhiG = np.linspace(0,np.pi/3,6)
fig = plt.figure(figsize=(20,10))
for ip,phiG in enumerate(listPhiG):
    args_fn = (sample,nShells,Vk,phiK,theta,w1p,w1d,w2p,w2d,stacking,Vg,phiG,kPts,range_k)
    fn = 'Data/ComparePhi/'+fsm.get_fn(*args_fn)+'.npz'
    if Path(fn).is_file():
        evals = np.load(fn)['evals']
        evecs = np.load(fn)['evecs']
    else:
        moire_pars = (Vg,Vk,phiG,phiK)
        args = (nShells, nCells, kList, monolayer_type, parsInterlayer, theta, moire_pars, '', False, True)
        evals, evecs = fsm.diagonalize_matrix(*args)
        np.savez(fn,evals=evals,evecs=evecs)
    #Compute weights
    weights = np.zeros((kPts,nCells*44))
    for i in range(kPts):
        ab = np.absolute(evecs[i])**2
        weights[i,:] = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    #Figure of points
    ax = fig.add_subplot(2,3,ip+1)
    for n in range(18*nCells,28*nCells):
        ax.plot(kLine,evals[:,n],color='r',lw=0.3,zorder=1)
        ax.scatter(kLine,evals[:,n], s=weights[:,n]*100,
                   color='b',lw=0,zorder=3)
    ax.set_ylim(-1.,-0.6)
    ax.set_xlim(-range_k,range_k)
    if ip in [0,3]:
        ax.set_ylabel("energy [eV]")
    ax.set_title(r"$\phi_\Gamma=$ %.2f °"%(phiG/np.pi*180),size=30)
plt.tight_layout()
plt.show()

