""" Plot of Figure 1: Bilayer with Interlayer coupling BUT without moirè.
Path Gamma -> K.
First ~8-10 bands.
"""

import numpy as np
import sys, os
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions as fs
from pathlib import Path
import matplotlib.pyplot as plt


machine = cfs.get_machine(os.getcwd())

sample = 'S11'
spread_E = 0.030        #eV

theta = 2.8

VG = 0.017       # eV           # Not needed here but for book-keeping
phiG = 175/180*np.pi        #rad    # Not needed here but for book-keeping
w1p = -1.66         # eV
w1d = 0.324         # eV
stacking = 'P'
w2p = w2d = 0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}

""" Other useless pars """
nShells = 0
nCells = 1
monolayer_type = 'fit'
Vk = Vg = phiG = phiK = 0
moire_pars = (Vg,Vk,phiG,phiK)

# Path G->K
kList = cfs.get_kList('Kp-G-K',11)
K0 = np.linalg.norm(kList[0])   #val of |K|
kPts = 200
kList = np.zeros((kPts,2))
kList[:,0] = np.linspace(0,K0,kPts)
""" Evals and weights """
args_e_data = (sample,nShells,monolayer_type,Vk,phiK,theta,stacking,w2p,w2d,phiG,kPts)
th_e_data_fn = 'Data/final_data_e_'+fs.get_fn(*args_e_data)+'.npz'
if not Path(th_e_data_fn).is_file():
    args = (nShells, nCells, kList, monolayer_type, parsInterlayer, theta, moire_pars, '', False, True)
    evals, evecs = fs.diagonalize_matrix(*args)
    np.savez(th_e_data_fn,evals=evals,evecs=evecs)
else:
    evals = np.load(th_e_data_fn)['evals']
    evecs = np.load(th_e_data_fn)['evecs']

if sample=='S11':
    pass
    #evals -= 0.47

""" Orbitals: d_xy, d_xz, d_z2, p_x, p_z """
orbitals = np.zeros((5,44,kPts))
list_orbs = ([6,7],[0,1],[5,],[3,4,9,10],[2,8])
for orb in range(5):
    inds_orb = list_orbs[orb]
    for il in range(2):     # 2 layers
        for ib in range(22):     #bands
            for ik in range(kPts):   #kpts
                for iorb in inds_orb:
                    orbitals[orb,il*22+ib,ik] += (
                        np.linalg.norm(evecs[ik,iorb,   il*22+ib])**2
                      + np.linalg.norm(evecs[ik,iorb+11,il*22+ib])**2
                      #+ np.linalg.norm(evecs[ik,iorb+22,il*22+ib])**2
                      #+ np.linalg.norm(evecs[ik,iorb+33,il*22+ib])**2
                    )

fig = plt.figure(figsize=(5,10))
ax = fig.add_subplot()
xvals = kList[:,0]
color = ['red','brown','blue','green','aqua']
for i in range(8):
    #ax.plot(xvals,evals[:,27-i],color='k')
    for orb in range(5):
        ax.scatter(xvals,evals[:,27-i],s=(orbitals[orb,27-i]*100),
                   marker='o',
                   facecolor=color[orb],
                   lw=0,
                   alpha=0.3,
                   )

ax.set_ylim(-2.5,0)
plt.show()










