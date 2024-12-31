import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions1 as fs
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D
from time import time as ttt
from datetime import timedelta


"""
Too see if the result is good we want to see:
    bands superimposed with experiment,
    orbital content,
    tb parameters change from DFT.
"""

machine = cfs.get_machine(os.getcwd())
ind_spec_args = 0 if len(sys.argv)==1 else int(sys.argv[1])
ind_random = 0 if len(sys.argv)<3 else int(sys.argv[2])
spec_args = fs.get_spec_args(ind_spec_args)
TMD = spec_args[0]
ind_reduced = spec_args[-1]
DFT_values = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,machine)
symm_data = fs.get_symm_data(exp_data)
reduced_data = fs.get_reduced_data(symm_data,ind_reduced)

print("------------CHOSEN PARAMETERS------------")
print(" TMD: ",spec_args[0],"\n chi2_1 parameter: ","{:.4f}".format(spec_args[1]),"\n Bound parameters: ","{:.2f}".format(spec_args[2]*100)+"%","\n Bounds SOC: ","{:.2f}".format(spec_args[3]*100)+"%","\n Index random evaluation: ",ind_random)
print(" Using 1 every ",ind_reduced," points, for a total of ",len(reduced_data[0])," points")

#Load result
SOC_fn = fs.get_SOC_fn(TMD,machine)
SOC_pars = np.load(SOC_fn)
temp_dn = fs.get_temp_dn(machine,spec_args)
min_chi2 = 1e5
for fn in os.listdir(temp_dn):
    if fn[-4:] == '.npy':
        temp_chi2 = float(fn.split('_')[-1][:-4])
        if temp_chi2<min_chi2:
            min_chi2 = temp_chi2
            min_fn = fn
temp_fn = temp_dn + min_fn
pars_tb = np.load(temp_fn)
full_pars = np.append(pars_tb,SOC_pars[-2:])

###################################################
#Superimposed bands
###################################################
fig_bands = temp_dn+'bands.png'
if not Path(fig_bands).is_file():
    fig = plt.figure(figsize=(20,14))
    ax = fig.add_subplot()
    KGK_end = exp_data[0][0][-1,0]
    KMKp_beg = exp_data[1][0][0,0]
    ikl = exp_data[0][0].shape[0]//2+1
    HSO = cfs.find_HSO(SOC_pars[-2:])
    tb_en = cfs.energy(full_pars,HSO,reduced_data,spec_args[0])
    tb_en2 = cfs.energy(DFT_values,cfs.find_HSO(DFT_values[-2:]),reduced_data,spec_args[0])
    for b in range(2):
    #        ax.plot(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
    #        ax.plot(exp_data[1][b][:,0]+KGK_end-KMKp_beg,exp_data[1][b][:,1],color='b',marker='*')
        ax.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='o',label='ARPES' if b == 0 else '',zorder=1,
                markersize=10,mew=1,mec='k',mfc='firebrick')
        ax.plot(reduced_data[b][:,0],tb_en2[b],color='g',marker='^',ls='-',label='DFT' if b == 0 else '',zorder=2,
                markersize=10,mew=1,mec='k',mfc='darkgreen')
        ax.plot(reduced_data[b][:,0],tb_en[b],color='skyblue',marker='s',ls='-',label='Fit' if b == 0 else '',zorder=3,
                markersize=10,mew=1,mec='k',mfc='deepskyblue')
    #
    ks = [reduced_data[0][0,0],4/3*np.pi/cfs.dic_params_a_mono[spec_args[0]],reduced_data[0][-1,0]]
    ax.set_xticks(ks,["$\Gamma$","$K$","$M$"],size=20)
    for i in range(3):
        ax.axvline(ks[i],color='k',lw=0.5)
    ax.set_xlim(reduced_data[0][0,0],reduced_data[0][-1,0])
#    ax.set_xlabel(r'$A^{-1}$',size=20)
    ax.set_ylabel('energy (eV)',size=30)
    label_y = []
    ticks_y = np.linspace(-0.4,-2,5)
    for i in ticks_y:
        label_y.append("{:.1f}".format(i))
    ax.set_yticks(ticks_y,label_y,size=20)
    plt.legend(fontsize=20)
    plt.savefig(fig_bands)
    plt.close()

##########################################################
#tb parameters change
##########################################################
fig_pars = temp_dn+'pars_distance.png'
if not Path(fig_pars).is_file():
    fig = plt.figure(figsize=(20,14))
    ax1 = fig.add_subplot()
    ax1.bar(np.arange(len(full_pars[:-3])),full_pars[:-3]-DFT_values[:-3],
            color='r',width=0.4,align='edge')
    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(full_pars[:-3])),(full_pars[:-3]-DFT_values[:-3])/abs(np.array(DFT_values[:-3]))*100,
            color='b',
            align='edge',width=-0.4
           )
    #ax1 
    ax1.set_ylim(-max(abs(full_pars[:-3]-DFT_values[:-3])),max(abs(full_pars[:-3]-DFT_values[:-3])))
    ax1.set_xticks(np.arange(len(full_pars[:-3])),cfs.list_formatted_names_all[:-3],size=15)
    ax1.tick_params(axis='y', labelsize=20,labelcolor='r')
    ax1.set_xlim(-0.5,len(full_pars[:-3])-0.5)
    #ax2
    ticks_y = np.linspace(-spec_args[2]*100,spec_args[2]*100,5)
    label_y = ["{:.1f}".format(i)+"\%" for i in ticks_y]
    ax2.set_yticks(ticks_y,label_y,size=20,color='b')
    ax2.set_ylim(-spec_args[2]*100,spec_args[2]*100)
    box_dic = dict(boxstyle='round',facecolor='none',alpha=0.5)
    ax2.fill_between([-0.5,6.5],[-spec_args[2]*100,-spec_args[2]*100],[spec_args[2]*100,spec_args[2]*100],alpha=.1,color='g',zorder=-1)
    ax2.text(0,-20,"Orbitals' on-site energy",size=20,color='g',bbox=box_dic)
    ax2.fill_between([6.5,35.5],[-spec_args[2]*100,-spec_args[2]*100],[spec_args[2]*100,spec_args[2]*100],alpha=.1,color='r',zorder=-1)
    if spec_args[0]=='WSe2':
        ax2.text(12,-20,"Hopping parameters",size=20,color='salmon',bbox=box_dic)
    else:
        ax2.text(15,-20,"Hopping parameters",size=20,color='salmon',bbox=box_dic)
    ax2.fill_between([35.5,40.5],[-spec_args[2]*100,-spec_args[2]*100],[spec_args[2]*100,spec_args[2]*100],alpha=.1,color='y',zorder=-1)
    ax2.text(36,-20,"Further\nneighbor\nhoppings",size=20,color='y',bbox=box_dic)
    #Legend
    handles = [
        Line2D([0,1], [0,1], color='r', linewidth=10, label="Parameter \'absolute\' difference"),
        Line2D([0,1], [0,1], color='b', linewidth=10, label="Parameter relative difference"),  ]
    ax2.legend(handles=handles,loc='upper left',fontsize=20,facecolor='w',framealpha=1)
    fig.tight_layout()
    plt.savefig(fig_pars)
    plt.close()

##############################################
# Orbital content image
##############################################

fig_orb = temp_dn+'orbital_content.png'
if not Path(fig_orb).is_file():
    fig,axs = plt.subplots(nrows=2,ncols=1,figsize=(15,10),gridspec_kw={'hspace':0,'right':0.877,'left':0.05,'top':0.98,'bottom':0.05})
    TMD = spec_args[0]
    a_TMD = cfs.dic_params_a_mono[TMD]
    for subp in range(2):   #DFT and fit orbitals
        par_values = np.array(cfs.initial_pt[TMD])  if subp == 0 else full_pars
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

        ax = axs[subp]
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
#            ax.xaxis.set_tick_params(labelsize=20)
    #
    #ax.set_title(txt_title,size=30)
    box_dic = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    axs[0].text(1.04,0.5,"DFT",size=30,bbox=box_dic,transform=axs[0].transAxes)
    axs[1].text(1.04,0.5,"Fit",size=30,bbox=box_dic,transform=axs[1].transAxes)
    plt.savefig(fig_orb)
    plt.close()










