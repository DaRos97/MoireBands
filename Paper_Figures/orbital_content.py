import numpy as np
import matplotlib.pyplot as plt
import sys,os
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
from pathlib import Path


fig = plt.figure(figsize=(20,10))
for TMD in cfs.TMDs:
    a_TMD = cfs.dic_params_a_mono[TMD]
    par_values = np.array(cfs.initial_pt[TMD])  #DFT values
    #G-K-M-G
    Nmg = 70
    N2 = 3
    Ngk = int(Nmg*2/np.sqrt(3))
    Nkm = int(Nmg*1/np.sqrt(3))
    Nk = Ngk+Nkm+Nmg+1  #+1 so we compute G twice
    K = np.array([4*np.pi/3/a_TMD,0])
    M = np.array([np.pi/a_TMD,np.pi/np.sqrt(3)/a_TMD])
    data = np.zeros((Nk,2))
    list_k = np.linspace(0,K[0],Ngk,endpoint=False)
    data[:Ngk,0] = list_k
    for ik in range(Nkm):
        data[Ngk+ik] = K + (M-K)/Nkm*ik
    for ik in range(Nmg):
        data[Ngk+Nkm+ik] = M - M/Nmg*ik
    #
    hopping = cfs.find_t(par_values)
    epsilon = cfs.find_e(par_values)
    offset = par_values[-3]
    #
    HSO = cfs.find_HSO(par_values[-2:])
    args_H = (hopping,epsilon,HSO,a_TMD,offset)
    #
    all_H = cfs.H_monolayer(data,*args_H)
    ens = np.zeros((Nk,22))
    evs = np.zeros((Nk,22,22),dtype=complex)
    for i in range(Nk):
        #index of TVB is 13, the other is 12 (out of 22: 11 bands times 2 for SOC. 7/11 are valence -> 14 is the TVB)
        ens[i],evs[i] = np.linalg.eigh(all_H[i])


    ax = fig.add_subplot(121) if TMD=='WSe2' else fig.add_subplot(122)
    color = ['g','','pink','m','','r','b','','pink','m','']
    marker = ['s','','o','s','','o','^','','o','s','']
    xvals = np.linspace(0,Nk-1,Nk)
    for i in range(22):
        ax.plot(xvals,ens[:,i],'k-',lw=0.3,zorder=0)
        for orb in [5,6,0]:    #3 different d orbitals
            for ko in range(0,Nk,N2):   #kpts
                orb_content = np.linalg.norm(evs[ko,orb,i])**2 + np.linalg.norm(evs[ko,orb+11,i])**2
                if orb in [6,0]:
                    orb_content += np.linalg.norm(evs[ko,orb+1,i])**2 + np.linalg.norm(evs[ko,orb+1+11,i])**2
                ax.scatter(xvals[ko],ens[ko,i],s=orb_content*100,edgecolor=color[orb],marker=marker[orb],facecolor='none',lw=2,zorder=1)
    xvals = np.linspace(Nk,2*Nk-1,Nk)
    for i in range(22):
        ax.plot(xvals,ens[:,i],'k-',lw=0.3,zorder=0)
        for orb in [2,3]:    #3 different d orbitals
            for ko in range(0,Nk,N2):   #kpts
                orb_content = np.linalg.norm(evs[ko,orb,i])**2 + np.linalg.norm(evs[ko,orb+11,i])**2
                orb_content += np.linalg.norm(evs[ko,orb+6,i])**2 + np.linalg.norm(evs[ko,orb+6+11,i])**2
                if orb in [3,]:
                    orb_content += np.linalg.norm(evs[ko,orb+1,i])**2 + np.linalg.norm(evs[ko,orb+1+11,i])**2
                    orb_content += np.linalg.norm(evs[ko,orb+6+1,i])**2 + np.linalg.norm(evs[ko,orb+1+6+11,i])**2
                ax.scatter(xvals[ko],ens[ko,i],s=orb_content*100,edgecolor=color[orb],marker=marker[orb],facecolor='none',lw=2,zorder=1)
    l_N = [0,Ngk,Ngk+Nkm,Nk]
    for l in range(3):
        mm = np.min(ens) -0.2
        MM = np.max(ens) +0.2
        for i in range(3):
            if l==2 and i==1:
                break
            ax.plot([l_N[i]+Nk*l,l_N[i]+Nk*l],[mm,MM],lw=0.5,color='k',zorder=0)
    #
    ax.set_xlim(0,2*Nk)
    ax.set_ylim(mm,MM)
    ax.set_xticks([0,Ngk,Ngk+Nkm,Nk,Nk+Ngk,Nk+Ngk+Nkm,2*Nk],[r'$\Gamma$',r'$K$',r'$M$',r'$\Gamma$',r'$K$',r'$M$',r'$\Gamma$'])
    ax.yaxis.set_tick_params(labelsize=15)
    ax.xaxis.set_tick_params(labelsize=20)
    if TMD=='WSe2':
        ax.set_ylabel("Energy (eV)",fontsize=20)
    #
        #Legend 1
        from matplotlib.lines import Line2D
        leg1 = []
        name = [r'$d_{xz}+d_{yz}$','',r'$p_z$',r'$p_x+p_y$','',r'$d_{z^2}$',r'$d_{xy}+d_{x^2-y^2}$']
        for i in [5,6,0]:
            leg1.append( Line2D([0], [0], marker=marker[i], markeredgecolor=color[i], markeredgewidth=2, label=name[i],
                                  markerfacecolor='none', markersize=10, lw=0)
                                  )
#        legend1 = ax.legend(handles=leg1,loc='upper left',bbox_to_anchor=(0.32,0.74),fontsize=15)
        legend1 = ax.legend(handles=leg1,loc='upper left',bbox_to_anchor=(0.99,1),fontsize=15)
        ax.add_artist(legend1)

        leg2 = []
        for i in [2,3]:
            leg2.append( Line2D([0], [0], marker=marker[i], markeredgecolor=color[i], markeredgewidth=2, label=name[i],
                                  markerfacecolor='none', markersize=10, lw=0)
                                  )
#        legend2 = ax.legend(handles=leg2,loc='upper left',bbox_to_anchor=(0.82,0.74),fontsize=15)
        legend2 = ax.legend(handles=leg2,loc='upper left',bbox_to_anchor=(1.018,0.8),fontsize=15)
        ax.add_artist(legend2)
    #   
        txt_title = r'WSe$_2$'
    else:
        txt_title = r'WS$_2$'
    ax.set_title(txt_title,size=30)
fig.tight_layout()
plt.subplots_adjust(wspace=0.239)
plt.show()
















