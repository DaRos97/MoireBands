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

""" Indices of parameters acording to type and orbital """
indOff = [40,]
indSOC = [41,42,]
indPz = [3,5,12,15,19,20,24,26,31,32,34,36,37,38]
indPxy = [4,6,13,14,16,17,25,27,33,35,39]

""" Indices of in and out-of -plane orbitals """
indOPO = [0,1,2,5,8,
          11,12,13,16,19]
indIPO = [3,4,6,7,9,10,
          14,15,17,18,20,21]
TVB2 = [12,13]
TVB4 = [10,11,12,13]
BCB2 = [14,15]

def get_args(ind):
    """ Function to retrieve arguments to pass to the minimization function. Mostly to decide constraint weights and parameters bounds. """
    lTMDs = ["WSe2", ]    #TMDs
    # Parameters of constraints
    lK1 = np.linspace(0.005,0.1,20)#[0,0.01,0.1,1]         # coefficient of parameters distance from DFT
    lK2 = [5,10]        # coefficient of band content
    lK3 = [20,]         # coefficient of minimum of conduction band
    lK4 = [1,]          # coefficient of gap value
    # Bounds
    lrp = [3,]         #tb bounds for general orbitals
    lrpz = [3,]         #tb bounds for z orbitals -> indices 6 and 9
    lrpxy = [3,]         #tb bounds for xy orbitals -> indices 7,8 and 10,11
    lrl = [3,]          #SOC bounds
    # Points in fit
    pts = [61,]     # better is it is a number n*3 + 1 with n integer
    listPar = list(itertools.product(*[lTMDs,lK1,lK2,lK3,lK4,lrp,lrpz,lrpxy,lrl,pts]))
    print("Index %d / %d"%(ind,len(listPar)))
    return listPar[ind]

def chi2_full(pars_full,*args):
    """
    Wrapper of chi2 with SOC parameters.
    """
    data,machine,args_minimization,max_eval = args
    SOC_pars = pars_full[-2:]
    HSO = cfs.find_HSO(SOC_pars)
    args_chi2 = (data,HSO,SOC_pars,machine,args_minimization,max_eval)
    pars_tb = pars_full[:-2]
    return chi2(pars_tb,*args_chi2)

def chi2(pars_tb,*args):
    """
    Compute square difference of bands with exp data.
    Made for fitting without SOC parameters -> HSO already computed.
    """
    data, HSO, SOC_pars, machine, args_minimization, max_eval = args
    K1,K2,K3,K4 = args_minimization[1:5]
    full_pars = np.append(pars_tb,SOC_pars)
    result = 0
    # chi2 of bands distance: compute energy of new pars
    tb_en, cond_en = cfs.energy(full_pars,HSO,data.fit_data,args_minimization[0],conduction=True)
    nbands = tb_en.shape[0]
    chi2_band_distance = 0
    for b in range(nbands):
        band_distance += np.sum(
            np.absolute(
                (tb_en[b]-data.fit_data[:,3+b])#[~np.isnan(data[:,3+b])]
            )**2)
    # K1: parameters distance
    K1_par_dis = compute_parameter_distance(pars_tb,args_minimization[0])
    # K2: orbital band content
    args_H_bc = (cfs.find_t(full_pars),cfs.find_e(full_pars),HSO,cfs.dic_params_a_mono[args_minimization[0]],full_pars[-3])
    k_pts = np.array([
        np.zeros(2),        #Gamma
        data.K,             #K
        data.M,             #M
    ])
    Ham_bc = cfs.H_monolayer(k_pts,*args_H_bc)
    ## Gamma
    evals_G,evecs_G = np.linalg.eigh(Ham_bc[0])
    K2_G = abs( 2 -
               np.sum(
                   np.absolute(
                       evecs_G[indOPO,TVB2]
                   )**2))
    ## K valence
    evals_K,evecs_K = np.linalg.eigh(Ham_bc[1])
    K2_Kval = abs( 2 -
               np.sum(
                   np.absolute(
                       evecs_K[indIPO,TVB2]
                   )**2))
    ## K conduction
    K2_Kcon = abs( 2 -
               np.sum(
                   np.absolute(
                       evecs_K[indOPO,BCB2]
                   )**2))
    ## M
    evals_M,evecs_M = np.linalg.eigh(Ham_bc[2])
    K2_M = abs( 4 -
               np.sum(
                   np.absolute(
                       evecs_M[indIPO,TVB4]
                   )**2))
    K2_band_content = K2_G + K2_Kval + K2_Kcon + K2_M
    if 0 :# chi2 of distance at Gamma and K and bands 1-2 point close to M -> specific for 10 pts
        chiDK = 0
        for i in range(2):  #2 bands
            indexes = [0,np.argmax(data[~np.isnan(data[:,3]),3])]    #indexes of Gamma (first element) and K /(highest energy)
            for j in range(2):  #Gamma and K
                chiDK += Pdk*(np.absolute(tb_en[i,indexes[j]]-data[indexes[j],3+i])**2)
            # Crossing before M -> only for WSe2
            if args_minimization[0]=="WSe2":        #Specific for points (40,15,10)
                chiDK += Pdk*(np.absolute(tb_en[1+i,-4]-data[-4,4+i])**2)
        result += chiDK
    # K3: minimum of conduction band
    if abs(data.fit_data[np.argmin(cond_en),0]-data.K[0])<1e-3:
        K3_band_min = 0
    else:
        K3_band_min = 1
    # K4: gap at K
    DFT_pars = np.array(cfs.initial_pt[args_minimization[0]])
    args_H_DFT = (cfs.find_t(DFT_pars),cfs.find_e(DFT_pars),cfs.find_HSO(DFT_pars[-2:]),cfs.dic_params_a_mono[args_minimization[0]],DFT_pars[-3])
    k_pts = np.array([ data.K, ])
    Ham_DFT = cfs.H_monolayer(k_pts,*args_H_bc)
    evals_DFT = np.linalg.evalsh(Ham_DFT)
    gap_DFT = evals_DFT[14]-evals_DFT[13]
    gap_p = evals_K[14]-evals_K[13]
    K4_gap = abs(gap_DFT-gap_p)

    result = chi2_band_distance + K1*K1_par_dis + K2*K2_band_content + K3*K3_band_min + K4*K4_gap

    ## From here on just plotting and temporary save
    #Save temporary file if result goes down
    home_dn = get_home_dn(machine)
    temp_dn = cfs.getFilename(('temp',*args_minimization),dirname=home_dn+'Data/')+'/'
    if not Path(temp_dn).is_dir():
        os.system("mkdir "+temp_dn)
    global min_chi2
    global evaluation_step
    evaluation_step += 1
    if result < min_chi2 and abs(result-min_chi2)>1e-4:   #remove old temp and add new one
        if not min_chi2==1e5:
            temp_fn = cfs.getFilename(('temp',min_chi2),dirname=temp_dn,extension='.npy')
            os.system('rm '+temp_fn)
        min_chi2 = result
        temp_fn = cfs.getFilename(('temp',result),dirname=temp_dn,extension='.npy')
        pars_full = np.append(pars_tb,SOC_pars)
        np.save(temp_fn,pars_full)
    if evaluation_step>max_eval:
        print("Reached max number of evaluations, plotting results")
        best_fn = cfs.getFilename(('temp',min_chi2),dirname=temp_dn,extension='.npy')
        best_tb = np.load(best_fn)
        best_en = cfs.energy(best_tb,HSO,data,args_minimization[0])
        plotResults(best_tb,best_en,data,args_minimization,machine,result)
        exit()
    if machine=='loc' and evaluation_step%1000==0:
        print("New intermediate figure")
        print("Chi2 contributions:")
        print("Band distance: %.6f"%band_distance)
        print("Pars distance: %.6f"%par_dis)
        print("Band content: %.6f"%band_content)
        print("Distance G and K: %.6f"%chiDK)
        print("Gap difference: %.6f"%gap_difference)
        print(np.array(cfs.initial_pt[args_minimization[0]])[-3:])
        print("-->")
        print(SOC_pars)
        plotResults(np.append(pars_tb,SOC_pars),tb_en,data,args_minimization,machine,result,dn='temp',show=False)
    return result

def plotResults(pars,ens,data,args_minimization,machine,result='',dn='',show=False,which='all'):
    if which=='all':
        plot1 = plot2 = plot3 = True
    else:
        plot1 = False
        plot2 = False
        plot3 = False
        if 'orb' in which:
            plot3 = True
        if 'pars' in which:
            plot2 = True
        if 'band' in which:
            plot1 = True
    home_dn = get_home_dn(machine)
    if dn=='temp':
        Dn = cfs.getFilename(('temp',*args_minimization),dirname=home_dn+'Data/')+'/'
    else:
        Dn = home_dn + 'Data/'
    if plot1:
        fig1 = cfs.getFilename(('bands',*args_minimization),dirname=Dn,extension='.png')
        plot_bands(ens,data,args_minimization=args_minimization,title="chi2: %.8f"%result,figname=fig1 if not show else '',show=False,TMD=args_minimization[0])
    if plot2:
        fig2 = cfs.getFilename(('pars',*args_minimization),dirname=Dn,extension='.png')
        plot_parameters(pars,args_minimization,title="chi2: %.8f"%result,figname=fig2 if not show else '',show=False)
    #
    if plot3:
        fig3 = cfs.getFilename(('orbitals',*args_minimization),dirname=Dn,extension='.png')
        plot_orbitalContent(pars,args_minimization[0],args_minimization=args_minimization,title="chi2: %.8f"%result,figname=fig3 if not show else '',show=show)

def plot_bands(tb_en,data,args_minimization=None,title='',figname='',show=False,TMD='WSe2'):
    DFT_pars = np.array(cfs.initial_pt[TMD])
    HSO_DFT = cfs.find_HSO(DFT_pars[-2:])
    DFT_en = cfs.energy(DFT_pars,HSO_DFT,data,TMD)
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot()
    for b in range(data.shape[1]-3):
        targ = np.argwhere(np.isfinite(data[:,3+b]))    #select only non-nan values
        xline = data[targ,0]
        ax.plot(xline,data[targ,3+b],color='r',marker='o',label='ARPES' if b == 0 else '',zorder=1,
                markersize=10,mew=1,mec='k',mfc='firebrick')
        ax.plot(xline,tb_en[b,targ],color='skyblue',marker='s',ls='-',label='Fit' if b == 0 else '',zorder=3,
                markersize=10,mew=1,mec='k',mfc='deepskyblue')
        if 0:
            ax.plot(xline,DFT_en[b,targ],color='g',marker='^',ls='-',label='DFT' if b == 0 else '',zorder=2,
                    markersize=10,mew=1,mec='k',mfc='darkgreen')
    #
    ks = [data[0,0],4/3*np.pi/cfs.dic_params_a_mono[TMD],data[-1,0]]
    ax.set_xticks(ks,[r"$\Gamma$",r"$K$",r"$M$"],size=20)
    for i in range(len(ks)):
        ax.axvline(ks[i],color='k',lw=0.5)
    ax.set_xlim(ks[0],ks[-1])
    ax.set_ylabel('energy (eV)',size=30)
    label_y = []
    if data.shape[1]==9:
        ticks_y = np.linspace(np.max(data[:,3])+0.2,np.min(data[~np.isnan(data[:,6]),6])-0.2,5)
        for i in ticks_y:
            label_y.append("{:.1f}".format(i))
        ax.set_yticks(ticks_y,label_y,size=20)
    plt.legend(fontsize=20)

    if not args_minimization is None:   #Additional text with parameters
        rp,rpz,rpxy,rl = args_minimization[5:9]
        box_dic = dict(boxstyle='round',facecolor='white',alpha=1)
        ax.text(
            0.05,0.85,
            "Bounds of parameters:\n"+"gen:  %d"%(rp*100)+"%\n"+"z:      %d"%(rpz*100) + "%\n"+"xy:    %d"%(rpxy*100) + "%\n"+"SOC:  %d"%(rl*100)+"%",
            bbox = box_dic,
            transform=ax.transAxes,
            fontsize=15
        )
        Ppar,Pbc,Pdk,Pgap = args_minimization[1:5]
        ax.text(
            0.3,0.83,
            "Chi2 parameters:\n"+"Ppar:  %.3f"%Ppar + "\n"+"Pbc:   %d"%Pbc + "\n"+"Pdk:   %d"%Pdk + "\n"+"Pgap:  %.3f"%Pgap,
            bbox = box_dic,
            transform=ax.transAxes,
            fontsize=15
        )
    #
    if not title=='':
        ax.set_title(title)
    if not figname=='':
        print("Saving figure: "+figname)
        plt.savefig(figname)
        plt.close(fig)
    if show:
        plt.show()

def plot_parameters(full_pars,args_minimization,title='',figname='',show=False):
    TMD = args_minimization[0]
    rp,rpz,rpxy,rl = args_minimization[5:9]
    rmax = max([rp,rpz,rpxy,rl])
    lenP = len(full_pars)
    #
    DFT_values = cfs.initial_pt[TMD]
    fig = plt.figure(figsize=(26,12))
    ax1 = fig.add_subplot()
    ax1.bar(np.arange(lenP),full_pars-DFT_values,
            color='r',width=0.4,align='edge',zorder=10)
    ax2 = ax1.twinx()
    ax2.bar(np.arange(lenP),(full_pars-DFT_values)/abs(np.array(DFT_values))*100,
            color='b',
            align='edge',width=-0.4,zorder=11
           )
    for ip in range(lenP):
        ax1.axvline(
            ip,
            ls='dashed',
            lw=0.8
        )
    #ax1 
    ax1.set_ylim(-max(abs(full_pars-DFT_values)),max(abs(full_pars-DFT_values)))
    ax1.set_ylabel("Absolute difference from DFT",size=18,color='r')
    ax1.set_xticks(np.arange(lenP),cfs.list_formatted_names_all,size=15)
    ax1.set_xlabel("Parameter name",size=18)
    ax1.tick_params(axis='y', labelsize=15,labelcolor='r')
    ax1.set_xlim(-0.5,lenP-0.5)
    top_ax = ax1.secondary_xaxis('top')
    top_ax.set_xticks(np.arange(lenP),["%d"%i for i in np.arange(lenP)],size=15)
    top_ax.set_xlabel("Parameter index",size=18)
    #ax2
    ticks_y = np.linspace(-rmax*100,rmax*100,5)
    label_y = ["{:.1f}".format(i)+r"%" for i in ticks_y]
    ax2.set_yticks(ticks_y,label_y,size=15,color='b')
    ax2.set_ylabel("Relative distance from DFT",size=18,color='b')
    ax2.set_ylim(-rmax*100,rmax*100)
    # Colors
    cp = "g"
    cpz = "r"
    cpxy = "navy"
    cpl = "aqua"
    cOff = "gold"
    for ip in range(lenP):
        if ip in indOff:
            c = cOff
        elif ip in indPz:
            c = cpz
        elif ip in indPxy:
            c = cpxy
        elif ip in indSOC:
            c = cpl
        else:
            c = cp
        ax2.fill_between([ip-0.5,ip+0.5],[-rmax*100,-rmax*100],[rmax*100,rmax*100],alpha=.3,color=c,zorder=-5,lw=0)
    box_dic = dict(boxstyle='round',facecolor='white',alpha=1)
    ax2.text(0,rmax*80,r"Parameters with $p_z$ and $d_{z^2}$: %d "%(int(rpz*100))+'%',size=20,color=cpz,bbox=box_dic)
    ax2.text(15,rmax*80,r"Parameters with $p_{xy}$ and $d_{x,y}$: %d "%(int(rpxy*100))+'%',size=20,color=cpxy,bbox=box_dic)
    ax2.text(27,rmax*80,r"Other parameters: %d "%(int(rp*100))+'%',size=20,color=cp,bbox=box_dic)
    ax2.text(36,rmax*80,r"Offset",size=20,color=cOff,bbox=box_dic)
    ax2.text(39,rmax*80,r"SOC: %d "%(int(rl*100))+'%',size=20,color=cpl,bbox=box_dic)
    if not title=='':
        ax1.set_title(title,size=20)
    if not figname=='':
        print("Saving figure: "+figname)
        plt.savefig(figname)
        plt.close(fig)
    if show:
        plt.show()

def plot_orbitalContent(full_pars,TMD,args_minimization=None,title='',figname='',show=False):
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
    hopping = cfs.find_t(full_pars)
    epsilon = cfs.find_e(full_pars)
    offset = full_pars[-3]
    #
    HSO = cfs.find_HSO(full_pars[-2:])
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
    if 0:       # Print orbitals at specific points
        vk = evs[Ngk]
        for ib in [12,13,14,15]:
            print("Band %d"%ib)
            v =vk[:,ib]
            for iorb in range(22):
                print("Orb #%d, %.4f"%(iorb,np.absolute(v[iorb])))
            print("--------------------------")
        print("Total values")
        print(orbitals[:,13,Ngk+Nkm])
        exit()
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

    if not args_minimization is None:   #Additional text with parameters
        rp,rpz,rpxy,rl = args_minimization[5:9]
        box_dic = dict(boxstyle='round',facecolor='white',alpha=1)
        ax.text(
            0.45,0.87,
            "Bounds of parameters:\n"+"gen:  %d"%(rp*100)+"%\n"+"z:      %d"%(rpz*100) + "%\n"+"xy:    %d"%(rpxy*100) + "%\n"+"SOC:  %d"%(rl*100)+"%",
            bbox = box_dic,
            transform=ax.transAxes,
            fontsize=15
        )
        Ppar,Pbc,Pdk,Pgap = args_minimization[1:5]
        ax.text(
            0.05,0.5,
            "Chi2 parameters:\n"+"Ppar:  %.3f"%Ppar + "\n"+"Pbc:   %d"%Pbc + "\n"+"Pdk:   %d"%Pdk + "\n"+"Pgap:  %.3f"%Pgap,
            bbox = box_dic,
            transform=ax.transAxes,
            fontsize=15
        )

    fig.tight_layout()

    if not title=='':
        ax.set_title(title,size=13)
    if figname!='':
        print("Saving figure: "+figname)
        fig.savefig(figname)
    if show:
        plt.show()
    plt.close()

def get_bounds(in_pt,args_minimization):
    rp, rpz, rpxy, rl = args_minimization[5:9]
    Bounds = []
    for i in range(in_pt.shape[0]):     #tb parameters
        if i == indOff: #offset
            temp = (-3,0)
            continue
        if i in indSOC: #SOC
            r = abs(rl*in_pt[i])
        elif i in indPz:
            r = rpz*abs(in_pt[i])
        elif i in indPxy:
            r = rpxy*abs(in_pt[i])
        else:
            r = rp*abs(in_pt[i])
        temp = (in_pt[i]-r,in_pt[i]+r)
        Bounds.append(temp)
    return Bounds

def compute_parameter_distance(pars,TMD):
    """
    Compue distance of current parameter set from DFT values.

    Returns
    -------
    float
    """
    DFT_values = np.array(cfs.initial_pt[TMD])
    len_tb = DFT_values.shape[0]
    if pars.shape[0]==len_tb:
        return np.sum(np.absolute((pars[:-3]-DFT_values[:-3])/DFT_values[:-3])) + np.sum(np.absolute((pars[-2:]-DFT_values[-2:])/DFT_values[-2:]))
    elif pars.shape[0]==len_tb-2:
        return np.sum(np.absolute((pars[:-1]-DFT_values[:-3])/DFT_values[:-3]))
    else:
        raise ValueError("compute_parameter_distance error")

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/Code/1_monolayer/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/1_monolayer/'
    elif machine == 'maf':
        return '/users/rossid/1_monolayer/'



































