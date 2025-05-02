import numpy as np
import sys,os,h5py
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D

TMD = 'WSe2'

"""
Image of bands.
"""
data_fn = 'data_figures/monolayer_bands_'+TMD+'.npy'
data = np.load(data_fn)
kline = data[0]
e_arpes = data[1:3]
e_fit = data[3:5]
e_dft = data[5:7]
exp_fn = 'data_figures/ARPES_moolayer_'+TMD+'.hdf5'
if Path(exp_fn).is_file():
    with h5py.File(exp_fn,'r') as f:
        es = np.copy(f['energies'])
        ks = np.copy(f['momenta'])
        intensity = np.copy(f['intensities'])
else:
    exp_fn2 = '../../Data_Experiments/Monolayers/KGM_'+TMD+'_triplet_cropped.txt'
    with open(exp_fn2,'r') as f:
        ls = f.readlines()
        ks = []
        for i in range(len(ls)):
            t = ls[i].split()
            k = float(t[0])
            e = float(t[1])
            if i==0:
                e0 = e
            if abs(e-e0)>1e-10:
                nk = i
                ks = np.array(ks)
                break
            else:
                ks.append(k)
        ne = len(ls)//nk
        es = np.zeros(ne)
        for i in range(ne):
            es[i] = float(ls[i*nk].split()[1])
        intensity = np.zeros((nk,ne))
        for i in range(len(ls)):
            intensity[i%nk,i//nk] = float(ls[i].split()[2])
    with h5py.File(exp_fn,'w') as f:
        f.create_dataset('energies',data=es)
        f.create_dataset('momenta',data=ks)
        f.create_dataset('intensities',data=intensity)

ne = es.shape[0]
nk = ks.shape[0]
em = es[0]; eM = es[-1]
km = ks[0]; kM = ks[-1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow((intensity.T[::-1,:]/255)**1,cmap='gray_r')
plt.show()
exit()
for b in range(2):
    ax.plot(np.arange(len(kline))/(len(kline)-1)*nk,(eM-e_fit[b])/(eM-em)*ne,color='r',ls='--',lw=0.4)
    ax.plot(np.arange(len(kline))/(len(kline)-1)*nk,(eM-e_dft[b])/(eM-em)*ne,color='g',ls=(0,(5,10)),lw=0.8)
ax.axvline(nk/3*2,color='k',lw=0.5)
ax.set_xticks([0,nk/3*2,nk],['$\Gamma$','$K$','$M$'],size=20)
ax.set_xlim(0,nk)
etick_vals = [0,-0.5,-1,-1.5,-2]
etick_lab = []
ie = []
for e in etick_vals:
    ie.append((eM-e)/(eM-em)*ne)
    etick_lab.append("{:.1f}".format(e))
ax.set_yticks(ie,etick_lab,size=20)
#
plt.show()
exit()
"""
Image of orbital content.
"""
fn = 'data_figures/orb_content_'+TMD+'.hdf5'
Nmg = 20
Ngk = int(Nmg*2/np.sqrt(3))
Nkm = int(Nmg*1/np.sqrt(3))
Nk = Ngk+Nkm+Nmg+1  #+1 so we compute G twice
N2 = 3
with h5py.File(fn,'r') as f:
    data_k = np.copy(f['k_points'])
    data_evals = np.copy(f['evals'])
    data_evecs = np.copy(f['evecs'])

for subp in range(2):   #DFT and fit orbitals
    ax = fig.add_subplot(2,2,2*(subp+1))
    color = ['g','','pink','m','','r','b','','pink','m','']
    marker = ['s','','o','s','','o','^','','o','s','']
    xvals = np.linspace(0,Nk-1,Nk)
    for i in range(22):
        ax.plot(xvals,data_evals[subp,:,i],'k-',lw=0.3,zorder=0)
        for orb in [5,6,0]:    #3 different d orbitals
            for ko in range(0,Nk,N2):   #kpts
                orb_content = np.linalg.norm(data_evecs[subp,ko,orb,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+11,i])**2
                if orb in [6,0]:
                    orb_content += np.linalg.norm(data_evecs[subp,ko,orb+1,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+1+11,i])**2
                ax.scatter(xvals[ko],data_evals[subp,ko,i],s=orb_content*100,edgecolor=color[orb],marker=marker[orb],facecolor='none',lw=2,zorder=1)
    xvals = np.linspace(Nk,2*Nk-1,Nk)
    for i in range(22):
        ax.plot(xvals,data_evals[subp,:,i],'k-',lw=0.3,zorder=0)
        for orb in [2,3]:    #3 different d orbitals
            for ko in range(0,Nk,N2):   #kpts
                orb_content = np.linalg.norm(data_evecs[subp,ko,orb,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+11,i])**2
                orb_content += np.linalg.norm(data_evecs[subp,ko,orb+6,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+6+11,i])**2
                if orb in [3,]:
                    orb_content += np.linalg.norm(data_evecs[subp,ko,orb+1,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+1+11,i])**2
                    orb_content += np.linalg.norm(data_evecs[subp,ko,orb+6+1,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+1+6+11,i])**2
                ax.scatter(xvals[ko],data_evals[subp,ko,i],s=orb_content*100,edgecolor=color[orb],marker=marker[orb],facecolor='none',lw=2,zorder=1)
    l_N = [0,Ngk,Ngk+Nkm,Nk]
    for l in range(3):
        mm = np.min(data_evals[subp]) -0.2
        MM = np.max(data_evals[subp]) +0.2
        for i in range(3):
            if l==2 and i==1:
                break
            ax.plot([l_N[i]+Nk*l,l_N[i]+Nk*l],[mm,MM],lw=0.5,color='k',zorder=0)
    #
    ax.set_xlim(0,2*Nk)
    ax.set_ylim(mm,MM)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_ylabel("Energy (eV)",fontsize=20)
    if subp==0:
        ax.set_xticks([])
        #Legend 1
        leg1 = []
        name = [r'$d_{xz}+d_{yz}$','',r'$p_z$',r'$p_x+p_y$','',r'$d_{z^2}$',r'$d_{xy}+d_{x^2-y^2}$']
        for i in [5,6,0]:
            leg1.append( Line2D([0], [0], marker=marker[i], markeredgecolor=color[i], markeredgewidth=2, label=name[i],
                                  markerfacecolor='none', markersize=10, lw=0)
                                  )
        legend1 = ax.legend(handles=leg1,loc=(1.003,0.01),
                            fontsize=20,handletextpad=0.35,handlelength=0.5)
        ax.add_artist(legend1)
        #Legend2
        leg2 = []
        for i in [2,3]:
            leg2.append( Line2D([0], [0], marker=marker[i], markeredgecolor=color[i], markeredgewidth=2, label=name[i],
                                  markerfacecolor='none', markersize=10, lw=0)
                                  )
        legend2 = ax.legend(handles=leg2,loc=(1.003,-0.2),
                            fontsize=20,handletextpad=0.35,handlelength=0.5)
        ax.add_artist(legend2)
    else:
        ax.set_xticks([0,Ngk,Ngk+Nkm,Nk,Nk+Ngk,Nk+Ngk+Nkm,2*Nk],[r'$\Gamma$',r'$K$',r'$M$',r'$\Gamma$',r'$K$',r'$M$',r'$\Gamma$'],size=20)


plt.show()

























