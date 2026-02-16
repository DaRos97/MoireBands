import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_monolayer as fsm
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
machine = cfs.get_machine(os.getcwd())          #Machine on which the computation is happening

if __name__ == "__main__":
    main()

def main():
    if len(sys.argv) != 2:
        print("Usage: py select_best_result.py arg1")
        print("arg1: WSe2 or WS2")
        exit()
    TMD = sys.argv[1]

    useDFT = False

    dataFn = "Figures/result_"+TMD+'.npy'
    if not useDFT:       # Use fit params
        par_values = np.load(dataFn)
    else:       # 
        par_values = cfs.initial_pt[TMD]
    plotOrbitalContent(parValues,TMD)

def plotOrbitalContent(par_values,TMD,figname='',show=False):
    """ Parameters cut: G-K-M-G """
    Ngk = 200
    Nkm = int(Ngk/2)
    Nmg = int(Ngk/2*np.sqrt(3))
    Nk = Ngk+Nkm+Nmg+1  #+1 so we compute G twice
    #
    a_TMD = cfs.dic_params_a_mono[TMD]
    K = np.array([4*np.pi/3/a_TMD,0])
    M = np.array([np.pi/a_TMD,np.pi/np.sqrt(3)/a_TMD])
    data_k = np.zeros((Nk,2))
    # G-K
    list_k = np.linspace(0,K[0],Ngk,endpoint=False)
    data_k[:Ngk,0] = list_k
    # K-M
    for ik in range(Nkm):
        data_k[Ngk+ik] = K + (M-K)/Nkm*ik
    # M-G
    for ik in range(Nmg+1):
        data_k[Ngk+Nkm+ik] = M + M/Nmg*ik
    """ Energies and evecs """
    hopping = cfs.find_t(par_values)
    epsilon = cfs.find_e(par_values)
    offset = par_values[-3]
    #
    HSO = cfs.find_HSO(par_values[-2:])
    args_H = (hopping,epsilon,HSO,a_TMD,offset)
    #
    all_H = cfs.H_monolayer(data_k,*args_H)
    ens = np.zeros((Nk,22))
    evs = np.zeros((Nk,22,22),dtype=complex)
    for i in range(Nk):
        #index of TVB is 13, the other is 12 (out of 22: 11 bands times 2 for SOC. 7/11 are valence -> 14 is the TVB)
        ens[i],evs[i] = np.linalg.eigh(all_H[i])
    """ Orbitals: d_xy, d_xz, d_z2, p_x, p_z """
    orbitals = np.zeros((5,22,Nk))
    list_orbs = ([6,7],[0,1],[5,],[3,4,9,10],[2,8])
    for orb in range(5):
        inds_orb = list_orbs[orb]
        for ib in range(22):     #bands
            for ik in range(Nk):   #kpts
                for iorb in inds_orb:
                    orbitals[orb,ib,ik] += np.linalg.norm(evs[ik,iorb,ib])**2 + np.linalg.norm(evs[ik,iorb+11,ib])**2
    if 0:       # Print orbitals at M
        vk = evs[Ngk+Nkm]
        for ib in [10,11,12,13]:
            print("Band %d"%ib)
            v =vk[:,ib]
            for iorb in range(22):
                print("Orb #%d, %.4f"%(iorb,np.absolute(v[iorb])))
            print("--------------------------")
        print("Total values")
        print(orbitals[:,13,Ngk+Nkm])
        #exit()
    """ Plot """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()

    color = ['red','brown','blue','green','aqua']
    labels = [r"$d_{xy}+d_{x^2-y^2}$",r"$d_{xz}+d_{yz}$",r"$d_{z^2}$",r"$p_x+p_y$",r"$p_z$"]

    leg = []
    xvals = np.linspace(0,Nk-1,Nk)
    for orb in range(5):
        for ib in range(22):
            ax.scatter(xvals,ens[:,ib],s=(orbitals[orb,ib]*100),
                       marker='o',
                       facecolor=color[orb],
                       lw=0,
                       alpha=0.3,
                       )
        leg.append( Line2D([0], [0], marker='o',
                           markeredgecolor='none',
                           markerfacecolor=color[orb],
                           markersize=10,
                           label=labels[orb],
                           lw=0)
                   )
    legend = ax.legend(handles=leg,
                       loc=(0.7,0.45),
                       fontsize=20,
                       handletextpad=0.35,
                       handlelength=0.5
                       )
    ax.add_artist(legend)

    ax.set_ylim(-4,3)
    ax.set_xlim(0,Nk-1)

    ax.axvline(Ngk,color='k',lw=1,zorder=-1)
    ax.axvline(Ngk+Nkm,color='k',lw=1,zorder=-1)
    ax.axhline(0,color='k',lw=1,zorder=-1)

    ax.set_xticks([0,Ngk-1,Ngk+Nkm-1,Nk-1],[r"$\Gamma$",r"$K$",r"$M$",r"$\Gamma$"],size=20)
    ax.set_ylabel("Energy [eV]",size=20)

    fig.tight_layout()

    if figname!='':
        print("Saving figure: "+figname)
        fig.savefig(figname)
    if show:
        plt.show()
    plt.close()






















































