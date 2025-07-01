import numpy as np
import CORE_functions as cfs
import scipy.linalg as la
from pathlib import Path
import sys,os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools

"""temp value of chi 2"""
global min_chi2
global evaluation_step
min_chi2 = 1e5
evaluation_step = 0

def get_spec_args(ind):
    lTMDs = cfs.TMDs    #TMDs
    lP = [0.01,0.05,0.07,0.1,0.2,0.3]         #coefficient of parameters distance from DFT chi2
    lrp = [0.1,0.2,0.3,0.5,0.7,1]         #tb bounds
    lrl = [0,0.1,0.2,0.3]          #SOC bounds
    lReduced = [13,]
    lPbc = [10,]        #coefficient of band content chi2
    lPdk = [20,]        #coefficient of distance at gamma and K chi2
    return list(itertools.product(*[lTMDs,lP,lrp,lrl,lReduced,lPbc,lPdk]))[ind]

def chi2_off_SOC(pars_SOC,*args):
    """
    Compute square difference of bands with exp data and compare G, K and maybe M point.
    """
    reduced_data, other_pars, TMD, machine = args
    HSO = cfs.find_HSO(pars_SOC[1:])
    full_pars = np.append(other_pars,pars_SOC)
    tb_en = cfs.energy(full_pars,HSO,reduced_data,TMD)
    #
    result = 0
    k_pts = len(reduced_data[0])
    plot = False
    if plot:
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1,1,1)
    for b in range(2):
        farg = np.zeros(4,dtype=int)
        farg[0] = np.argmax(reduced_data[b][:k_pts//4,1])   #Max at Gamma
        farg[1] = np.argmin(reduced_data[b][:k_pts//2,1])   #Min between Gamma and K
        farg[2] = k_pts//2+np.argmax(reduced_data[b][k_pts//2:3*k_pts//4,1])    #max at K
        farg[3] = k_pts-2   #Min at M, -1 because M might be a bit weird
        #
        if plot:
            ax.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='*',label='new symm' if b == 0 else '')
            ax.plot(reduced_data[b][farg,0],tb_en[b,farg],color='g',marker='^',ls='-',label='fit' if b == 0 else '')
        #
        for i in [0,2]:  #just the maxima at gamma and K
            increase = np.absolute(tb_en[b,farg[i]]-reduced_data[b][farg[i],1])
            result += increase
            if plot:
                print(np.absolute(tb_en[b,farg[i]]-reduced_data[b][farg[i],1]))
                ax.scatter(reduced_data[b][farg[i],0],reduced_data[b][farg[i],1],c='k',marker='*',zorder=10,s=200)
    if plot:
        plt.show()
    return result

def chi2_full(pars_full,*args):
    """
    Wrapper of chi2_tb with SOC paameters.
    """
    reduced_data,machine,spec_args,ind_random,max_eval = args
    SOC_pars = pars_full[-2:]
    HSO = cfs.find_HSO(SOC_pars)
    args_chi2 = (reduced_data,HSO,SOC_pars,machine,spec_args,ind_random,max_eval)
    pars_tb = pars_full[:-2]
    return chi2_tb(pars_tb,*args_chi2)

def chi2_tb(pars_tb,*args):
    """
    Compute square difference of bands with exp data.
    Made for fitting WITHOUT SOC parameters -> HSO already computed.
    """
    reduced_data, HSO, SOC_pars, machine, spec_args, ind_random, max_eval = args
    full_pars = np.append(pars_tb,SOC_pars)
    #Compute energy of new pars
    tb_en = cfs.energy(full_pars,HSO,reduced_data,spec_args[0])
    #
    result = 0
    #chi2 of bands distance
    for b in range(2):
        result += np.sum(np.absolute((tb_en[b]-reduced_data[b][:,1])[~np.isnan(reduced_data[b][:,1])])**2)
    #chi2 of parameters distance
    par_dis = compute_parameter_distance(pars_tb,spec_args[0])
    result += spec_args[1]*par_dis
    #chi2 of bands' content
    band_content = np.array(compute_band_content(full_pars,HSO,spec_args[0]))
    Pbc = spec_args[5]
    result += Pbc*(2-np.sum(np.absolute(band_content)**2))
    #chi2 of distance at Gamma and K
    Pdk = spec_args[6]
    for i in range(2):  #2 bands
        indexes = [0,np.argmax(reduced_data[0][~np.isnan(reduced_data[i][:,1]),1])]    #indexes of Gamma and K for ind_reduced=14      #####
        for j in range(2):  #Gamma and K
            result += Pdk*(np.absolute(tb_en[i,indexes[j]]-reduced_data[i][indexes[j],1])**2)
    #Save temporary file if result goes down
    global min_chi2
    if result < min_chi2 and ind_random >= 0:   #remove old temp and add new one
        temp_fn = get_temp_fit_fn(min_chi2,spec_args,ind_random,machine)
        if not min_chi2==1e5:
            os.system('rm '+temp_fn)
        min_chi2 = result
        temp_fn = get_temp_fit_fn(min_chi2,spec_args,ind_random,machine)
        pars_full = np.append(pars_tb,SOC_pars)
        np.save(temp_fn,pars_full)
    global evaluation_step
    evaluation_step += 1
    if evaluation_step>max_eval:
        print("reached max number of evaluations")
        exit()
    #Plot figure every N steps to see how it is going
    if machine=='loc':    #Plot each nnnn steps
        nnnn = 1000
        if evaluation_step%nnnn==0:
            plot_bands(tb_en,reduced_data,title="chi2: "+"{:.4f}".format(result),figname='Figures/temp.png',show=False)
            print("New fig ",evaluation_step//nnnn,", chi2: ","{:.8f}".format(result))
            if 0:#fig of distance from DFT values
                fig = plt.figure(figsize=(15,20))
                ax1 = fig.add_subplot(2,1,1)
                ax1.bar(np.arange(len(pars_tb)),pars_tb-cfs.initial_pt[spec_args[0]][:-2],color='r')
                ax1.set_ylabel("Absolute")
                ax2 = fig.add_subplot(2,1,2)
                ax2.bar(np.arange(len(pars_tb)),(pars_tb-cfs.initial_pt[spec_args[0]][:-2])/abs(np.array(cfs.initial_pt[spec_args[0]][:-2]))*100,color='b')
                ax2.set_ylabel("Percentage")
                ax1.set_title("chi2: "+"{:.4f}".format(result))
                fig.tight_layout()
                plt.savefig('Figures/memp.png')
                plt.close(fig)
    #print("chi2: ","{:.7f}".format(result))
    return result

def plot_bands(tb_en,reduced_data,dft_en=np.zeros(0),title='',figname='',show=False,TMD='WSe2'):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot()
    for b in range(2):
        targ = np.argwhere(np.isfinite(reduced_data[b][:,1]))    #select only non-nan values
        xline = reduced_data[b][targ,0]
        ax.plot(xline,reduced_data[b][targ,1],color='r',marker='o',label='ARPES' if b == 0 else '',zorder=1,
                markersize=10,mew=1,mec='k',mfc='firebrick')
#        ax.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='*',label='new symm' if b == 0 else '')
        ax.plot(xline,tb_en[b,targ],color='skyblue',marker='s',ls='-',label='Fit' if b == 0 else '',zorder=3,
                markersize=10,mew=1,mec='k',mfc='deepskyblue')
#        ax.plot(reduced_data[b][targ,0],tb_en[b,targ],color='g',marker='^',ls='-',label='fit' if b == 0 else '')
        if not dft_en.shape[0]==0:
            ax.plot(xline,dft_en[b,targ],color='g',marker='^',ls='-',label='DFT' if b == 0 else '',zorder=2,
                    markersize=10,mew=1,mec='k',mfc='darkgreen')
    #
    ks = [xline[0][0],4/3*np.pi/cfs.dic_params_a_mono[TMD],xline[-1][0]]
    ax.set_xticks(ks,[r"$\Gamma$",r"$K$",r"$M$"],size=20)
    for i in range(3):
        ax.axvline(ks[i],color='k',lw=0.5)
    ax.set_xlim(ks[0],ks[-1])
    ax.set_ylabel('energy (eV)',size=30)
    label_y = []
    ticks_y = np.linspace(np.max(tb_en)+0.2,np.min(tb_en)-0.2,5)
    for i in ticks_y:
        label_y.append("{:.1f}".format(i))
    ax.set_yticks(ticks_y,label_y,size=20)
    plt.legend(fontsize=20)
    #
    if not title=='':
        ax.set_title(title)
    if not figname=='':
        plt.savefig(figname)
        plt.close(fig)
    if show:
        plt.show()

def plot_parameters(full_pars,spec_args,title='',figname='',show=False):
    TMD = spec_args[0]
    DFT_values = cfs.initial_pt[TMD]
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
    label_y = ["{:.1f}".format(i)+r"%" for i in ticks_y]
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
    if not title=='':
        ax1.set_title(title)
    if not figname=='':
        plt.savefig(figname)
        plt.close(fig)
    if show:
        plt.show()

def plot_orbitals(full_pars,title='',figname='',show=False,TMD='WSe2'):
    Ngk = 16
    Nkm = Ngk//2#int(Nmg*1/np.sqrt(3))
    Nk = Ngk+Nkm+1  #+1 so we compute G twice
    N2 = 2
    #
    a_TMD = cfs.dic_params_a_mono[TMD]
    K = np.array([4*np.pi/3/a_TMD,0])
    M = np.array([np.pi/a_TMD,np.pi/np.sqrt(3)/a_TMD])
    data_k = np.zeros((Nk,2))
    list_k = np.linspace(0,K[0],Ngk,endpoint=False)
    data_k[:Ngk,0] = list_k
    for ik in range(Nkm+1):
        data_k[Ngk+ik] = K + (M-K)/Nkm*ik
    data_evals = np.zeros((2,Nk,22))
    data_evecs = np.zeros((2,Nk,22,22),dtype=complex)
    for p in range(2):
        par_values = np.array(cfs.initial_pt[TMD])  if p == 0 else full_pars
        #
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
        data_evals[p] = ens
        data_evecs[p] = evs
    #Actual plot
    fig,axs = plt.subplots(nrows=2,ncols=1,figsize=(15,10),gridspec_kw={'hspace':0,'right':0.877,'left':0.05,'top':0.98,'bottom':0.05})
    for subp in range(2):   #DFT and fit orbitals
        ax = axs[subp]
        color = ['g','','pink','m','','r','b','','pink','m','']
        marker = ['s','','o','s','','o','^','','o','s','']
        #d orbitals
        xvals = np.linspace(0,Nk-1,Nk)
        for i in range(22):
            ax.plot(xvals,data_evals[subp,:,i],'k-',lw=0.3,zorder=0)
            for orb in [5,6,0]:    #3 different d orbitals
                for ko in range(0,Nk,N2):   #kpts
                    orb_content = np.linalg.norm(data_evecs[subp,ko,orb,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+11,i])**2
                    if orb in [6,0]:
                        orb_content += np.linalg.norm(data_evecs[subp,ko,orb+1,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+1+11,i])**2
                    ax.scatter(xvals[ko],data_evals[subp,ko,i],s=orb_content*100,edgecolor=color[orb],marker=marker[orb],facecolor='none',lw=2,zorder=1)
        #p orbitals
        xvals = np.linspace(Nk-1,2*Nk-2,Nk)
        for i in range(22):
            ax.plot(xvals,data_evals[subp,::-1,i],'k-',lw=0.3,zorder=0)
            for orb in [2,3]:    #3 different d orbitals
                for ko in range(Nk-1,-1,-N2):   #kpts
                    orb_content = np.linalg.norm(data_evecs[subp,ko,orb,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+11,i])**2
                    orb_content += np.linalg.norm(data_evecs[subp,ko,orb+6,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+6+11,i])**2
                    if orb in [3,]:
                        orb_content += np.linalg.norm(data_evecs[subp,ko,orb+1,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+1+11,i])**2
                        orb_content += np.linalg.norm(data_evecs[subp,ko,orb+6+1,i])**2 + np.linalg.norm(data_evecs[subp,ko,orb+1+6+11,i])**2
                    ax.scatter(xvals[ko],data_evals[subp,Nk-1-ko,i],s=orb_content*100,edgecolor=color[orb],marker=marker[orb],facecolor='none',lw=2,zorder=1)
        l_N = [0,Ngk,Ngk+Nkm,Ngk+Nkm+Nkm,2*Nk-2]
        for l in range(len(l_N)):
            ax.axvline(l_N[l],lw=0.5,color='k',zorder=0)
            mm = np.min(data_evals[subp]) -0.2
            MM = np.max(data_evals[subp]) +0.2
            continue
            for i in range(3):
                if l==2 and i==1:
                    break
                ax.plot([l_N[i]+Nk*l,l_N[i]+Nk*l],[mm,MM],lw=0.5,color='k',zorder=0)
        #
        ax.set_xlim(0,2*Nk-2)
        ax.set_ylim(mm,MM)
        ax.yaxis.set_tick_params(labelsize=15)
    #        ax.set_ylabel("Energy (eV)",fontsize=20)
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
            ax.set_xticks(l_N,[r'$\Gamma$',r'$K$',r'$M$',r'$K$',r'$\Gamma$'],size=20)
    box_dic = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    axs[0].text(1.04,0.5,"DFT",size=30,bbox=box_dic,transform=axs[0].transAxes)
    axs[1].text(1.04,0.5,"Fit",size=30,bbox=box_dic,transform=axs[1].transAxes)
    if not title=='':
        axs[0].set_title(title)
    if not figname=='':
        plt.savefig(figname)
        plt.close(fig)
    if show:
        plt.show()

def compute_band_content(parameters,HSO,TMD):
    """
    Computes the band content of d0 and p0e at Gamma and the band content of dp2 and ppe at K.
    Returns a list with the coefficients: (c1t,c1,c6t,c6) in the notation of Fang et al., which
    are the components of d0,p0e,dp2,ppe respectvely.
    HSO not really needed but time not so critical here.
    """
    functions_kpt = [[d0,p0e],[dp2,ppe]]
    args_H = (cfs.find_t(parameters),cfs.find_e(parameters),HSO,cfs.dic_params_a_mono[TMD],parameters[-3])
    k_pts = np.array([
        np.zeros(2),        #Gamma
        cfs.R_z(np.pi/3)@np.array([4/3*np.pi/cfs.dic_params_a_mono[TMD],0])     #K
    ])
    H = cfs.H_monolayer(k_pts,*args_H)
    result = []
    for i in range(k_pts.shape[0]): #kpt
        evals,evecs = np.linalg.eigh(H[i,:11,:11])
        for j in range(2):  #2 orbitals at each momentum point
            result.append(functions_kpt[i][j](evecs[:,6]))  # 6 -> top valence band
    return result

def get_exp_data(TMD,machine):
    """
    For given material, takes the two cuts (KGK and KMKp) and the two bands and returns the lists of energy and momentum for the 2 top valence bands.
    We take from the .txt experiment data which has value of |k| and energy and save it as a .npy matrix with values:
        |k|, energy, kx, ky.
    Need to handle some NANs in the energy -> that point is not available -> still keep it.
    """
    data = []
    offset_exp = {
        'WSe2':{'KGK':0,'KMKp':-0.0521},
        'WS2':{'KGK':0,'KMKp':-0.0025}
    } #To align the two cuts -> fixed on symmetrized data
    for cut in ['KGK','KMKp']:
        data.append([])
        for band in range(1,3):
            data_fn = get_exp_data_fn(TMD,cut,band,machine)     #.npy file saved
            if Path(data_fn).is_file():
                data[-1].append(np.load(data_fn))
                continue
            with open(get_exp_fn(TMD,cut,band,machine), 'r') as f:  #original .txt file
                lines = f.readlines()
            temp = []
            for i in range(len(lines)):
                ke = lines[i].split('\t')       #momentum modulus and energy
                if ke[1] == 'NAN\n':
                    temp.append([float(ke[0]),np.nan,*find_vec_k(float(ke[0]),cut,TMD)])
                else:
                    temp.append([float(ke[0]),float(ke[1])+offset_exp[TMD][cut],*find_vec_k(float(ke[0]),cut,TMD)])
            data[-1].append(np.array(temp))
            np.save(data_fn,np.array(temp))
    return data

def get_symm_data(exp_data):
    """
    Symmetrize experimental data from k to -k, for the 2 cuts and the 2 bands.
    Experimental values of |k| are symmetric around 0, so each point has a symmetric one.
    We average between the two. If one of them is nan keep only the other. If both are nan give nan.
    We also put the data on a line G-K-M so the result is a 2xN matrix for the 2 bands.
    """
    Nkgk = len(exp_data[0][0])
    Nkmk = len(exp_data[1][0])
    symm_data = np.zeros((2,Nkgk//2+Nkmk//2+Nkmk%2,4))
    for i in range(2):  #two bands
        for ik in range(Nkgk//2,Nkgk):       #second half for kgk
            if np.isnan(exp_data[0][i][ik][1]):
                symm_data[i,ik-Nkgk//2] = exp_data[0][i][Nkgk-1-ik]
            else:
                if np.isnan(exp_data[0][i][Nkgk-1-ik][1]):
                    symm_data[i][ik-Nkgk//2] = exp_data[0][i][ik]
                else:       #actuall average
                    symm_data[i,ik-Nkgk//2] = np.array([exp_data[0][i][ik][0],(exp_data[0][i][ik][1]+exp_data[0][i][Nkgk-1-ik][1])/2,exp_data[0][i][ik][2],exp_data[0][i][ik][3]])
        for ik in range(Nkmk//2+Nkmk%2):       #first half for kmk
            if np.isnan(exp_data[1][i][ik][1]):
                symm_data[i,Nkgk//2+ik] = exp_data[1][i][Nkmk-1-ik]
            else:
                if np.isnan(exp_data[1][i][Nkmk-1-ik][1]):
                    symm_data[i][ik+Nkgk//2] = exp_data[1][i][ik]
                else:       #actuall average
                    symm_data[i,ik+Nkgk//2] = np.array([exp_data[1][i][ik][0],(exp_data[1][i][ik][1]+exp_data[1][i][Nkmk-1-ik][1])/2,exp_data[1][i][ik][2],exp_data[1][i][ik][3]])
        symm_data[i,Nkgk//2:,0] += symm_data[i,Nkgk//2-1,0] + exp_data[1][i][-1][0]
    return symm_data

def get_reduced_data(symm_data,ind):
    """
    Get reduced set of k-points for the comparison.
    """
    red_data = []
    for i in range(2):
        red_data.append(np.concatenate((symm_data[i][::ind],[symm_data[i][-1],]),axis=0))
    return red_data

def symmetrize(dataset):
    """dataset has N k-entries, each containing a couple (k,E,kx,ky)"""
    new_ds = []
    len_ds = len(dataset)//2 if len(dataset)%2 == 0 else len(dataset)//2+1
    for i in range(len_ds):
        temp = np.zeros(4)
        temp[0] = np.abs(dataset[i,0])#np.sqrt(dataset[i,2]**2+dataset[i,3]**2)
        temp[2:] = dataset[i,2:]
        if np.isnan(dataset[i,1]) and np.isnan(dataset[-1-i,1]):
            temp[1] = np.nan
        elif np.isnan(dataset[i,1]):
            temp[1] = dataset[-1-i,1]
        elif np.isnan(dataset[-1-i,1]):
            temp[1] = dataset[i,1]
        else:
#            temp[1] = (dataset[i,1]+dataset[-1-i,1])/2
            temp[1] = dataset[i,1]
        new_ds.append(temp)
    return np.array(new_ds)

def get_bounds(in_pt,spec_args):
    TMD, P, rp, rl, ind_reduced, Pbc, Pdk = spec_args
    Bounds = []
    for i in range(in_pt.shape[0]):     #tb parameters
        if i == in_pt.shape[0]-3: #offset
            temp = (-3,0)
        elif i == in_pt.shape[0]-2 or i == in_pt.shape[0]-1: #SOC
            r = rl*in_pt[i]
            temp = (in_pt[i]-r,in_pt[i]+r)
        else:
            r = rp*abs(in_pt[i])
            temp = (in_pt[i]-r,in_pt[i]+r)
        Bounds.append(temp)
    return Bounds

def find_vec_k(k_scalar,cut,TMD):
    """
    Compute vector components from the (signed)modulus depending cut and TMD.
    """
    a_mono = cfs.dic_params_a_mono[TMD]
    k_pts = np.zeros(2)
    if cut == 'KGK':
        k_pts[0] = k_scalar
        k_pts[1] = 0
    elif cut == 'KMKp':
        M = np.array([np.pi,np.pi/np.sqrt(3)])/a_mono
        K = np.array([4*np.pi/3,0])/a_mono
        Kp = np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])/a_mono
        if k_scalar < 0:
            k_pts = M + (K-M)*np.abs(k_scalar)/la.norm(K-M)
        else:
            k_pts = M + (Kp-M)*np.abs(k_scalar)/la.norm(Kp-M)
    return k_pts

def get_spec_args_txt(spec_args):
    return spec_args[0]+'_'+"{:.3f}".format(spec_args[1]).replace('.',',')+'_'+"{:.3f}".format(spec_args[2]).replace('.',',')+'_'+"{:.3f}".format(spec_args[3]).replace('.',',')+'_'+str(spec_args[4])+'_'+str(spec_args[5])+'_'+str(spec_args[6])

def get_exp_data_fn(TMD,cut,band,machine):
    return get_exp_dn(machine)+'extracted_data_'+cut+'_'+TMD+'_band'+str(band)+'.npy'

def get_exp_fn(TMD,cut,band,machine):
    return get_exp_dn(machine)+cut+'_'+TMD+'_band'+str(band)+'.txt'

def get_temp_fit_fn(chi,spec_args,ind_random,machine):
    return get_temp_dn(machine,spec_args)+'temp_'+str(ind_random)+"_"+"{:.8f}".format(chi)+'.npy'

def get_res_fn(TMD,machine):
    return get_fig_dn(machine)+'result_'+TMD+'.npy'

def get_fig_fn(spec_args,machine):
    return get_fig_dn(machine)+'fig_'+get_spec_args_txt(spec_args)+'.png'

def get_SOC_fn(TMD,machine):
    return get_res_dn(machine)+TMD+'_SOC.npy'

def get_exp_dn(machine):
    return get_home_dn(machine)+'inputs/'

def get_res_dn(machine):
    return get_home_dn(machine)+'Data/'

def get_fig_dn(machine):
    return get_home_dn(machine)+'Figures/'

def get_temp_dn(machine,spec_args):
    return get_res_dn(machine)+'temp_'+get_spec_args_txt(spec_args)+'/'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/Code/1_monolayer/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/1_monolayer/'
    elif machine == 'maf':
        return '/users/rossid/1_monolayer/'

def compute_parameter_distance(pars,TMD):
    DFT_values = np.array(cfs.initial_pt[TMD])
    len_tb = DFT_values.shape[0]
    if pars.shape[0]==len_tb:
        return np.sum(np.absolute(pars[:-3]-DFT_values[:-3])**2) + np.sum(np.absolute(pars[-2:]-DFT[-2:])**2)
    elif pars.shape[0]==len_tb-2:
        return np.sum(np.absolute(pars-DFT_values[:-2])**2)
    else:
        print("compute_parameter_distance error")

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################

def ppe(a):
    return -1/np.sqrt(2)*(a[9]+1j*a[10])
def ppo(a):
    return -1/np.sqrt(2)*(a[3]+1j*a[4])
def pme(a):
    return  1/np.sqrt(2)*(a[9]-1j*a[10])
def pmo(a):
    return  1/np.sqrt(2)*(a[3]-1j*a[4])
def p0e(a):
    return a[8]
def p0o(a):
    return a[2]
def d0(a):
    return a[5]
def dp1(a):
    return -1/np.sqrt(2)*(a[0]+1j*a[1])
def dm1(a):
    return  1/np.sqrt(2)*(a[0]-1j*a[1])
def dp2(a):
    return  1/np.sqrt(2)*(a[7]+1j*a[6])
def dm2(a):
    return  1/np.sqrt(2)*(a[7]-1j*a[6])

orb_txt = ['dxz','dyz','poz','pox','poy','dz2','dxy','dx2','pez','pex','pey']

def get_orbital_content(spec_args,machine,fn=''):
    print("_____________________________________________________________________________")
    print("Orbital content:")
    a_mono = cfs.dic_params_a_mono[spec_args[0]]
    k_pts = np.array([np.zeros(2),np.matmul(cfs.R_z(np.pi/3),np.array([4/3*np.pi/a_mono,0]))])    #Gamma and K (K+ of Fange et al., 2015)
    txt_pt = ['Gamma:','K:    ']
    fun_pt = [[d0,p0e],[dp2,ppe]]
    txt_fun_pt = [['d0 ','p0e'],['dp2','ppe']]
    #
    if fn=='':
        fn = get_res_fn(spec_args,machine)
    full_pars = np.load(fn)
    DFT_pars = np.array(cfs.initial_pt[spec_args[0]])
    #
    args_DFT = (cfs.find_t(DFT_pars),cfs.find_e(DFT_pars),cfs.find_HSO(DFT_pars[-2:]),a_mono,DFT_pars[-3])
    H_DFT = cfs.H_monolayer(k_pts,*args_DFT)
    args_res = (cfs.find_t(full_pars),cfs.find_e(full_pars),cfs.find_HSO(full_pars[-2:]),a_mono,full_pars[-3])
    H_res = cfs.H_monolayer(k_pts,*args_res)
    #
    print("      \tDFT values\t\tres values")
    for i in range(2):
        H0 = H_DFT[i,11:,11:]    #Spinless Hamiltonian
        H1 = H_res[i,11:,11:]    #Spinless Hamiltonian
        k = k_pts[i]
        E0,evec0 = np.linalg.eigh(H0)
        E1,evec1 = np.linalg.eigh(H1)
        a0 = evec0[:,6]
        a1 = evec1[:,6]
        #
        v0 = [fun_pt[i][0](a0),fun_pt[i][1](a0)]
        v1 = [fun_pt[i][0](a1),fun_pt[i][1](a1)]
        print(txt_pt[i]+'\t'+txt_fun_pt[i][0]+':    '+"{:.3f}".format(np.absolute(v0[0]))+'\t\t'+txt_fun_pt[i][0]+':    '+"{:.3f}".format(np.absolute(v1[0])))
        print('      '+'\t'+txt_fun_pt[i][1]+':    '+"{:.3f}".format(np.absolute(v0[1]))+'\t\t'+txt_fun_pt[i][1]+':    '+"{:.3f}".format(np.absolute(v1[1])))
        print('      '+'\tweight: '+"{:.3f}".format(np.absolute(v0[0])**2+np.absolute(v0[1])**2)+'\t\tweight: '+"{:.3f}".format(np.absolute(v1[0])**2+np.absolute(v1[1])**2))
        if i == 0:
            print("________________________________________________________")

def get_table(spec_args,machine,fn=''):
    print("_____________________________________________________________________________")
    print("Table of parameters with distance from DFT")
    if fn=='':
        fn = get_res_fn(spec_args,machine)
    full_pars = np.load(fn)
    pars_dft = cfs.initial_pt[spec_args[0]]
    list_names = cfs.list_names_all
    for i in range(len(full_pars)):
        percentage = np.abs((full_pars[i]-pars_dft[i])/pars_dft[i]*100)
        l = 10 - len(list_names[i])
        sp1 = '' if percentage>10 else ' '
        sp2 = '' if pars_dft[i]<0 else ' '
        sp3 = '' if full_pars[i]<0 else ' '
        print(list_names[i],':',' '*l,sp2,"{:.5f}".format(pars_dft[i]),'  ->  ',sp3,"{:.5f}".format(full_pars[i]),'    ',sp1,"{:.2f}".format(percentage),'%')






