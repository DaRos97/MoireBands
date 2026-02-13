import numpy as np
import matplotlib.pyplot as plt

fn = 'dataGM/fig2_withInterlayer.npz'       #Change filename to your location

""" There are 400 points from Gamma to K to K': kPts=200 is K
Chose a final point you prefer between 200 (K-point) and 400 (K'-point) """
kPts = 210

""" Extraction of data: sample 11 has an energy shift of 0.47 eV -> to match monolayer energies just shift them all by this amount. """
evals = np.load(fn)['evals']
evecs = np.load(fn)['evecs']
norm = np.load(fn)['norm']
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

""" Figure """
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
xvals = norm[:kPts]
color = ['red','brown','blue','green','aqua']
colorTop = ['red',]*5
colorBottom = ['aqua',]*5
for i in range(8):
    #ax.plot(xvals,evals[:,27-i],color='k')
    for orb in range(5):
        ax.scatter(xvals,evals[:kPts,27-i],s=(orbitals[orb,27-i]*100),
                   marker='o',
                   facecolor=color[orb],
                   lw=0,
                   alpha=0.3,
                   )
        ax2.scatter(xvals,evals[:kPts,27-i],s=(orbitalsTop[orb,27-i]*100),
                   marker='o',
                   facecolor=colorTop[orb],
                   lw=0,
                   alpha=0.1,
                   zorder=1,
                   )
        ax2.scatter(xvals,evals[:kPts,27-i],s=(orbitalsBottom[orb,27-i]*100),
                   marker='o',
                   facecolor=colorBottom[orb],
                   lw=0,
                   alpha=0.1,
                   zorder=0,
                   )

ax.set_ylim(-2.5,0)
ax2.set_ylim(-2.5,0)
plt.show()
