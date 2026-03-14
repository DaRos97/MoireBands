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
save = True

sample = 'S11'
spread_E = 0.030        #eV

theta = 2.8

VG = 0.017       # eV           # Not needed here but for book-keeping
phiG = 175/180*np.pi        #rad    # Not needed here but for book-keeping
w1p = 0#-1.66         # eV
w1d = 0#0.324         # eV
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
kPts = 400
kList,norm = cfs.get_kList('G-K-Kp',kPts,returnNorm=True)
K0 = np.linalg.norm(kList[-1])   #val of |K|
#kList = np.zeros((kPts,2))
#kList[:,0] = np.linspace(0,K0,kPts)
""" Evals and weights """
args_e_data = (sample,nShells,monolayer_type,Vk,phiK,theta,stacking,w1p,w1d,phiG,kPts)
th_e_data_fn = 'Data/final_data_e_'+fs.get_fn(*args_e_data)+'.npz'
if not Path(th_e_data_fn).is_file():
    print("Computing")
    args = (nShells, nCells, kList, monolayer_type, parsInterlayer, theta, moire_pars, '', False, True)
    evals, evecs = fs.diagonalize_matrix(*args)
    if save:
        np.savez(th_e_data_fn,evals=evals,evecs=evecs,norm=norm)
else:
    print("Loading")
    evals = np.load(th_e_data_fn)['evals']
    evecs = np.load(th_e_data_fn)['evecs']
    norm = np.load(th_e_data_fn)['norm']

if sample=='S11':
    pass
    #evals -= 0.47

kPts = evals.shape[0]
""" Orbitals: d_xy, d_xz, d_z2, p_x, p_z """
orbitals = np.zeros((5,44,kPts))
orbitalsTop = np.zeros((5,44,kPts))
orbitalsBottom = np.zeros((5,44,kPts))
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
                      + np.linalg.norm(evecs[ik,iorb+22,il*22+ib])**2  #bottom orbitals
                      + np.linalg.norm(evecs[ik,iorb+33,il*22+ib])**2  #bottom orbitals
                    )
                    orbitalsTop[orb,il*22+ib,ik] += (
                        np.linalg.norm(evecs[ik,iorb,   il*22+ib])**2
                      + np.linalg.norm(evecs[ik,iorb+11,il*22+ib])**2
                    )
                    orbitalsBottom[orb,il*22+ib,ik] += (
                        np.linalg.norm(evecs[ik,iorb+22,il*22+ib])**2
                      + np.linalg.norm(evecs[ik,iorb+33,il*22+ib])**2
                    )

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
xvals = np.linspace(0,2*K0,kPts)
color = ['red','brown','blue','green','aqua']
colorTop = ['red',]*5
colorBottom = ['powderblue',]*5
for i in range(8):
    #ax.plot(xvals,evals[:,27-i],color='k')
    for orb in range(5):
        ax.scatter(xvals,evals[:,27-i],s=(orbitals[orb,27-i]*100),
                   marker='o',
                   facecolor=color[orb],
                   lw=0,
                   alpha=0.3,
                   )
        ax2.scatter(xvals,evals[:,27-i],s=(orbitalsTop[orb,27-i]*100),
                   marker='o',
                   facecolor=colorTop[orb],
                   lw=0,
                   alpha=0.1,
                   zorder=1,
                   )
        ax2.scatter(xvals,evals[:,27-i],s=(orbitalsBottom[orb,27-i]*100),
                   marker='o',
                   facecolor=colorBottom[orb],
                   lw=0,
                   alpha=0.1,
                   zorder=0,
                   )

ax.set_ylim(-2.5,0)
ax2.set_ylim(-2.5,0)
plt.show()










