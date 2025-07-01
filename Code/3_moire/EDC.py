"""
Here we try to extract the moire potential at Gamma or K by looking at the distance between the main band and
the X shape of the side bands below.
This analysis is wrt S11, so at theta=2.8°.
We do this by only varying the Vg from 0.005 to 0.02, computing at k=Gamma (K) the distance between the main band
and the lower side band (need to get the weights, at Gamma (K) there is only one that remains nonzero). Finally,
we plot this distance as function of Vg and compare with experiment.
The observed distance in S11 is ~90 meV for Gamma and ~170 meV for K.
We also compare with different rotation angles to see the differences.

This script is made to be used together with edc.sh -> computes many instances of phi_G/K
And also with Compare_EDC.py which plots the trend of best_V with the phase and some error bars.
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
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from time import time
from tqdm import tqdm
machine = cfs.get_machine(cwd)
disp = True

if len(sys.argv)!=3:
    print("Usage: py EDC.py arg1 arg2")
    print("arg1: 'G' or 'K'\narg2: int from 0 to 30 -> 0 to 2*pi")
    exit()
else:
    momentum_point = sys.argv[1]
    phase = int(sys.argv[2])/30*np.pi*2

plot = 0#True
savefig = 0#True
#Parameters of calculation
monolayer_type, interlayer_symmetry, Vg, Vk, phiG, phiK, theta_0, sample, N, cut, k_pts, weight_exponent = fsm.get_pars(0)

if momentum_point=='G':
    phiG = phase
elif momentum_point=='K':
    phiK = phase
else:
    print("Error in input")
    exit()
exp_distance_X = 0.09 if momentum_point=='G' else 0.17
Ntheta = 3
list_V = np.linspace(0.0001,0.030,15) if momentum_point=='G' else np.linspace(0.0001,0.030,11)
list_V = [0.007,]

K_point = np.array([0,0]) if momentum_point=='G' else np.array([4/3*np.pi/cfs.dic_params_a_mono['WSe2'],0])
list_theta = np.linspace(theta_0-0.2,theta_0+0.2,Ntheta)
if disp:
    print("-----------PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Symmetry of interlayer coupling: ",interlayer_symmetry," with values from sample ",sample)
    if momentum_point=='G':
        print("Moiré potential values (eV,deg): G->(?,"+"{:.1f}".format(phiG/np.pi*180)+"°), K->("
              +"{:.4f}".format(Vk)+","+"{:.1f}".format(phiK/np.pi*180)+"°)")
    else:
        print("Moiré potential values (eV,deg): G->("+"{:.4f}".format(Vg)+","+"{:.1f}".format(phiG/np.pi*180)+"°), K->(?,"+"{:.1f}".format(phiK/np.pi*180)+"°)")
    print("Twist angle from "+"{:.2f}".format(list_theta[0])+"° to "+"{:.2f}".format(list_theta[-1])+"°.")
    print("Number of mini-BZs circles: ",N)
#Monolayer parameters
pars_monolayer = fsm.import_monolayer_parameters(monolayer_type,machine)
#Interlayer parameters
pars_interlayer = [interlayer_symmetry,np.load(fsm.get_pars_interlayer_fn(sample,interlayer_symmetry,monolayer_type,machine))]

"""
Calculation
"""
if plot:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0,1,Ntheta))
    colors[Ntheta//2] = (0,0,0,1)
#Indices of main band (MB) and of lower band (LB) -> model specific
ind_MB = 22 #index of main band of the layer, just half of 44 -> central mBZ of each layer
ind_LB = 26 if momentum_point=='G' else 27 #index of LB to consider. At G the SOC is degenerate
ind_UB = 28 #the valence band is 7/11 -> 28/44
def fit_func(x,a,b,c,d):
    """Function for fitting distance with V."""
    return a + b*x + c*x**2 + d*x**3

best_V = np.zeros(Ntheta)
long_list_V = np.linspace(list_V[0],list_V[-1],1000)

for ind_th in [1,]:#range(Ntheta):
    theta = list_theta[ind_th]
    if disp and 0:
        print("Theta: ",theta)
    distances_X = np.zeros(len(list_V))
    for iV in range(len(list_V)):
        V = list_V[iV]
        if disp and 1:
            print("V_"+momentum_point+" = ","{:.4f}".format(V))
        #Moire parameters
        moire_potentials = (V,Vk,phiG,phiK) if momentum_point=='G' else (Vg,V,phiG,phiK)
        pars_moire = fsm.import_moire_parameters(N,moire_potentials,theta)
        look_up = fsm.lu_table(pars_moire[0])
        energies = np.zeros(pars_moire[1]*44)
        weights = np.zeros(pars_moire[1]*44)
        H_tot = fsm.big_H(K_point,look_up,pars_monolayer,pars_interlayer,pars_moire)
        energies,evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
        ab = np.absolute(evecs)**2
        weights = np.sum(ab[:ind_MB,:],axis=0) + np.sum(ab[ind_MB*pars_moire[1]:ind_MB*pars_moire[1]+ind_MB,:],axis=0)
        inds = np.argsort(weights[pars_moire[1]*ind_LB:pars_moire[1]*ind_UB])
        ee = energies[pars_moire[1]*ind_LB+inds]
        distances_X[iV] = ee[-2]-ee[-3] if momentum_point=='G' else ee[-1]-ee[-2]   #again the SOC degeneracy

        if 1 and ind_th==1:   #plot image to see what's happening
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot()
#            ax.set_title("Phase "+"{:.3f}".format(phase)+", distance: "+"{:.5f}".format(distances_X[iV]))
            energies = np.zeros((k_pts,pars_moire[1]*44))
            weights = np.zeros((k_pts,pars_moire[1]*44))
            K_list = cfs.get_K(cut,k_pts)
            for i in tqdm(range(k_pts)):
                K_i = K_list[i]
                H_tot = fsm.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
                energies[i,:],evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
                ab = np.absolute(evecs)**2
                weights[i,:] = np.sum(ab[:ind_MB,:],axis=0) + np.sum(ab[ind_MB*pars_moire[1]:ind_MB*pars_moire[1]+ind_MB,:],axis=0)
            for e in range(10*pars_moire[1]):
                color = 'r'
                ax.plot(np.arange(k_pts),
                        energies[:,18*pars_moire[1]+e],
                        color=color,
                        lw=0.05,
                        zorder=2
                        )
                color = 'b'
                ax.scatter(np.arange(k_pts),
                        energies[:,18*pars_moire[1]+e],
                        s=weights[:,18*pars_moire[1]+e]**(0.5)*50,
                        lw=0,
                        color=color,
                        zorder=3
                        )
#            ax.set_ylabel("energy(eV)",size=20)
#            ax.yaxis.set_tick_params(labelsize=20)
            ax.set_xlim(200,300)
            ax.set_xticks([])
            ax.set_yticks([])
            #plot distace main band to X
            if momentum_point=='G':
                #ax.set_ylim(-1.5,-0.9)
                ax.set_ylim(-1.35,-1.1)
                indG = np.argwhere(np.linalg.norm(K_list,axis=1)<1e-7)[0]
                ax.plot([indG,indG],[ee[-2],ee[-3]],color='r',marker='o')
            else:
                ax.set_ylim(-1.3,-0.8)
                indK = np.argwhere(np.linalg.norm(K_list-K_point,axis=1)<1e-7)[0]
                ax.plot([indK,indK],[ee[-1],ee[-2]],color='r',marker='o')
            fig.tight_layout()
            plt.show()
            exit()

    #Select V for theta 2.8° with a curve fit
    popt, pcov = scipy.optimize.curve_fit(fit_func,list_V,distances_X)
    best_V[ind_th] = long_list_V[np.argmin(np.absolute(fit_func(long_list_V,*popt)-exp_distance_X))]

    if plot:
        ax.plot(list_V,distances_X,color=colors[ind_th])

if plot:        #Make image pretty
    ax.axhline(y=exp_distance_X,color='g')
    txt_label = r"$V_\Gamma$" if momentum_point=='G' else r"$V_K$"
    ax.set_xlabel(txt_label+' (eV)',size=20)
    txt_k = r"$\Gamma$" if momentum_point=='G' else r"$K$"
    ax.set_ylabel(r"Energy distance at "+txt_k+" between main band and EDC (eV)",size=20)
    title = r'EDC at $\Gamma$' if momentum_point=='G' else r'EDC at $K$'
    ax.set_title(title,size=25)
    norm = Normalize(vmin=list_theta[0]-theta_0,vmax=list_theta[-1]-theta_0)
    sm = ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm,ax=ax)
    cbar.set_label(r"Twist angle $\theta$ (deg) from 2.8°",size=20)
    #
    ax.set_ylim(ax.get_ylim())
    ax.set_xlim(list_V[0],list_V[-1])
    ax.plot([best_V[Ntheta//2],best_V[Ntheta//2]],ax.get_ylim(),color='k',ls='--')
    #
    ax.text(0.6,0.1,r"$\phi_"+momentum_point+"$="+"{:.3f}".format(phase)+'\nBest V: '+"{:.6f}".format(best_V[Ntheta//2]),transform=ax.transAxes,size=20)
    fig.tight_layout()
    if savefig:
        print("Saving figure and best V")
        figname = 'Figures/EDC_'+momentum_point+'_phi'+"{:.3f}".format(phase)+'.png'
        plt.savefig(figname)
        data_fn = 'Data/EDC_'+momentum_point+'_phi'+"{:.3f}".format(phase)+'.npy'
        np.save(data_fn,best_V)
    else:
        plt.show()

