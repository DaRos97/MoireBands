import numpy as np
import CORE_functions as cfs
import scipy.linalg as la
from pathlib import Path
import sys,os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import itertools
import copy

"""temp value of chi 2"""
global min_chi2
global evaluation_step
min_chi2 = 1e5
evaluation_step = 0

""" Indices of parameters acording to type and orbital """
indOff = [40,]
indSOC = [41,42,]
# Orbital indices
indPz = [3,5,12,15,19,20,24,26,31,32,34,36,37,38]
indPxy = [4,6,13,14,16,17,25,27,33,35,39]
# Type of parameter indices
indEps = list(np.arange(7))
indT1 = list(np.arange(7,28))
indT5 = list(np.arange(28,36))
indT6 = list(np.arange(36,40))

""" Indices of orbitals """
xz_i = 0
yz_i = 1
zo_i = 2
xo_i = 3
yo_i = 4
z2_i = 5
xy_i = 6
x2_i = 7
ze_i = 8
xe_i = 9
ye_i = 10
indOPO = [          # Out-of-plane orbitals
    xz_i,yz_i,zo_i,z2_i,ze_i,
    xz_i+11,yz_i+11,zo_i+11,z2_i+11,ze_i+11
]
indIPO = [          # In-plane orbitals
    xo_i,yo_i,xy_i,x2_i,xe_i,ye_i,
    xo_i+11,yo_i+11,xy_i+11,x2_i+11,xe_i+11,ye_i+11,
]
indILC = [          # Interlayer-coupling orbitals
    zo_i,z2_i,ze_i,
    zo_i+11,z2_i+11,ze_i+11,
]
TVB2 = [12,13]
TVB4 = [10,11,12,13]
BCB2 = [14,15]

""" Values of orbital character """
# Mod square of eigenvectors (plus SOC couple)
# For G: pze and dz2 (same for the 2 TVB which are degenerate)
# For K: p_-1^e tvb1, tvb2, d_2 tvb1, tvb2
orbital_character = {
    'WSe2': {
        'G': (0.2740,0.6606),    # Spinless is [0.2931,0.7069] (0.5414 is the square root -> same as Fang et al.),
        'K': (0.1856,0.2116,0.8144,0.7763)  # Spinless is [0.1980,0.8020] (0.4450 is the square root -> same as Fang et al.),
    },
    'WS2': {
        'G': (0.3205,0.6571),    # Spinless is [0.3282,0.6718] (0.5729 is the square root -> same as Fang et al.),
        'K': (0.1960,0.2366,0.8040,0.7575)  # Spinless is [0.2156,0.7844] (0.4643 is the square root -> same as Fang et al.),
    },
}

""" Args and chi2 function """
def get_args(TMD,ind):
    """
    Function to retrieve arguments to pass to the minimization function.
    Mostly to decide constraint weights and parameters bounds.

    Returns
    -------
    dict: 'pts', 'Ks', 'Bs'
    """
    # Parameters of constraints
    lK1 = [0,1e-5]         # coefficient of parameters distance from DFT
    lK2 = np.logspace(-7,1,10,base=2)        # coefficient of band content at M
    lK3 = np.logspace(-7,1,10,base=2)               # coefficient of orbital content of G and K from DFT
    lK4 = [0.5,1,]         # coefficient of minimum of conduction band at K
    lK5 = [0.1,0.5]          # coefficient of gap value
    lK6 = [1,5]             # weight of high symmetry points: G,K,near-M-crossing and M
    # Bounds
    boundType = 'absoute'      # or 'relative'
    if boundType=='relative':
        Bs = (10,10,10,0)       # pgen, pz, pxy, psoc
    elif boundType=='absolute':
        Bs = (5,2,4,1,0)          # peps, pt1, pt5, pt6, psoc. 0 means out of minimization
    # Points in fit
    pts = 91     # better is it is a number n*3 + 1 with n integer
    listPar = list(itertools.product(*[lK1,lK2,lK3,lK4,lK5,lK6]))
    print("Index %d / %d"%(ind,len(listPar)))
    listPar = listPar[ind]
    args = {
        'TMD': TMD,
        'pts': pts,
        'Ks':  tuple(listPar),
        'boundType': boundType,
        'Bs':  Bs,
    }
    return args

def chi2_full(pars_full,*args):
    """
    Wrapper of chi2 with SOC parameters.
    """
    data,machine,args_minimization,max_eval,returnEnergy = args
    SOC_pars = pars_full[-2:]
    HSO = cfs.find_HSO(SOC_pars)
    args_chi2 = (data,HSO,SOC_pars,machine,args_minimization,max_eval,returnEnergy)
    pars_tb = pars_full[:-2]
    return chi2(pars_tb,*args_chi2)

def chi2(pars_tb,*args):
    """
    Compute square difference of bands with exp data.
    Made for fitting without SOC parameters -> HSO already computed.
    """
    data, HSO, SOC_pars, machine, args_minimization, max_eval, returnEnergy = args
    K1,K2,K3,K4,K5,K6 = args_minimization['Ks']
    full_pars = np.append(pars_tb,SOC_pars)
    result = 0
    # chi2 of bands distance: compute energy of new pars -> and K5
    tb_en, cond_en = cfs.energy(full_pars,HSO,data.fit_data,args_minimization['TMD'],conduction=True)
    if returnEnergy:
        return tb_en
    nbands = tb_en.shape[0]
    chi2_band_distance = 0
    specialIndices = [0,np.argmax(data.fit_data[:,3]),np.argmin(data.fit_data[:,4]),data.fit_data.shape[0]-1]
    weights = np.ones(data.fit_data.shape[0])
    weights[specialIndices] = K6
    for ib in range(nbands):
        chi2_band_distance += np.sum(
            np.absolute(
                ((tb_en[ib]-data.fit_data[:,3+ib])*weights)[~np.isnan(data.fit_data[:,3+ib])]
            )**2) / data.fit_data[~np.isnan(data.fit_data[:,3+ib])].shape[0]
    # K1: parameters distance
    K1_par_dis = compute_parameter_distance(full_pars,args_minimization['TMD'])
    # K2: orbital band content
    args_H_bc = (cfs.find_t(full_pars),cfs.find_e(full_pars),HSO,cfs.dic_params_a_mono[args_minimization['TMD']],full_pars[-3])
    k_pts = np.array([
        data.M,             #M
        np.zeros(2),        #Gamma
        data.K,             #K
    ])
    Ham_bc = cfs.H_monolayer(k_pts,*args_H_bc)
    ## M
    evals_M,evecs_M = np.linalg.eigh(Ham_bc[0])
    bandsM = TVB4 if args_minimization['TMD']=='WSe2' else TVB2
    K2_M = np.sum( np.absolute( evecs_M[indILC,:][:,bandsM] )**2 )
    if args_minimization['TMD']=='WS2':
        K2_M *= 2       # to have same dimension to WSe2
    # K3: occupation at G and K
    ## Gamma
    evals_G,evecs_G = np.linalg.eigh(Ham_bc[1])
    occ_ze, occ_z2 = orbital_character[args_minimization['TMD']]['G']
    G_ze_tvb1 = np.absolute(evecs_G[ze_i,13])**2 + np.absolute(evecs_G[ze_i+11,13])**2
    G_ze_tvb2 = np.absolute(evecs_G[ze_i,12])**2 + np.absolute(evecs_G[ze_i+11,12])**2
    G_z2_tvb1 = np.absolute(evecs_G[z2_i,13])**2 + np.absolute(evecs_G[z2_i+11,13])**2
    G_z2_tvb2 = np.absolute(evecs_G[z2_i,12])**2 + np.absolute(evecs_G[z2_i+11,12])**2
    #
    evals_K,evecs_K = np.linalg.eigh(Ham_bc[2])
    occ_p1_tvb1, occ_p1_tvb2, occ_d2_tvb1, occ_d2_tvb2 = orbital_character[args_minimization['TMD']]['K']
    K_p1_tvb1 = (np.absolute(-1/np.sqrt(2)*(evecs_K[xe_i,13]-1j*evecs_K[ye_i,13]))**2
                 + np.absolute(-1/np.sqrt(2)*(evecs_K[xe_i+11,13]-1j*evecs_K[ye_i+11,13]))**2)
    K_p1_tvb2 = (np.absolute(-1/np.sqrt(2)*(evecs_K[xe_i,12]-1j*evecs_K[ye_i,12]))**2
                 + np.absolute(-1/np.sqrt(2)*(evecs_K[xe_i+11,12]-1j*evecs_K[ye_i+11,12]))**2)
    K_d2_tvb1 = (np.absolute(1/np.sqrt(2)*(evecs_K[x2_i,13]-1j*evecs_K[xy_i,13]))**2
                 + np.absolute(1/np.sqrt(2)*(evecs_K[x2_i+11,13]-1j*evecs_K[xy_i+11,13]))**2)
    K_d2_tvb2 = (np.absolute(1/np.sqrt(2)*(evecs_K[x2_i,12]-1j*evecs_K[xy_i,12]))**2
                 + np.absolute(1/np.sqrt(2)*(evecs_K[x2_i+11,12]-1j*evecs_K[xy_i+11,12]))**2)
    K3_DFT = (
        abs(occ_ze-G_ze_tvb1) +
        abs(occ_ze-G_ze_tvb2) +
        abs(occ_z2-G_z2_tvb1) +
        abs(occ_z2-G_z2_tvb2) +
        abs(occ_p1_tvb1-K_p1_tvb1) +
        abs(occ_p1_tvb2-K_p1_tvb2) +
        abs(occ_d2_tvb1-K_d2_tvb1) +
        abs(occ_d2_tvb2-K_d2_tvb2)
    )
    # K4: minimum of conduction band
    if abs(data.fit_data[np.argmin(cond_en),0]-data.K[0])<1e-3:
        K4_band_min = 0
    else:
        K4_band_min = 1
    # K5: gap at K
    DFT_pars = np.array(cfs.initial_pt[args_minimization['TMD']])
    args_H_DFT = (cfs.find_t(DFT_pars),cfs.find_e(DFT_pars),cfs.find_HSO(DFT_pars[-2:]),cfs.dic_params_a_mono[args_minimization['TMD']],DFT_pars[-3])
    k_pts = np.array([ data.K, ])
    Ham_DFT = cfs.H_monolayer(k_pts,*args_H_DFT)
    evals_DFT = np.linalg.eigvalsh(Ham_DFT)[0]
    gap_DFT = evals_DFT[14]-evals_DFT[13]
    gap_p = evals_K[14]-evals_K[13]
    K5_gap = abs(gap_DFT-gap_p)
    # Total result
    result = chi2_band_distance + (
        K1*K1_par_dis +
        K2*K2_M +
        K3*K3_DFT +
        K4*K4_band_min +
        K5*K5_gap
    )

    chi2_elements = (chi2_band_distance,K1_par_dis,K2_M,K3_DFT,K4_band_min,K5_gap)
    """ From here on just plotting and temporary save """
    home_dn = get_home_dn(machine)
    temp_fn = cfs.getFilename(
        ('res',args_minimization['TMD'],args_minimization['Ks'],args_minimization['boundType'],args_minimization['Bs']),
        dirname=home_dn+'Data/',
        floatPrecision=10,
        extension='.npz'
    )
    global min_chi2
    global evaluation_step
    evaluation_step += 1
    if result < min_chi2:
        min_chi2 = result
        np.savez(
            temp_fn,
            elements=chi2_elements,
            pars=full_pars
        )
    if evaluation_step>max_eval:
        print("Reached max number of evaluations, exiting minimization with result %.4f"%min_chi2)
        exit()
    if machine=='loc' and evaluation_step%1000==1:
        print("New intermediate figure")
        print("Result: %.6f"%result)
        print("Chi2: %.6f"%chi2_band_distance)
        print("---------------")
        print("1->par: %.6f"%(K1*K1_par_dis))
        print("---------------")
        print("2->orb: %.6f"%(K2*K2_M))
        print("---------------")
        print("3->orb: %.6f"%(K3*K3_DFT))
        print("---------------")
        print("4->min: %.6f"%(K4*K4_band_min))
        print("---------------")
        print("5->gap: %.6f"%(K5*K5_gap))
    return result

""" Plotting """
def plotResults(pars,TMD,Ks,boundType,Bs,chi2_elements,pts=91):
    pts = 91
    cwd = os.getcwd()
    master_folder = cwd[:40]
    data = cfs.monolayerData(TMD,master_folder,pts=pts)
    args_minimization = {
        'TMD': TMD,
        'pts': pts,
        'Ks': list(Ks),
        'boundType':boundType,
        'Bs': list(Bs)
    }
    args_chi2 = (data,'loc',args_minimization,1e5,True)
    tb_en = chi2_full(pars,*args_chi2)
    legendInfo = (TMD,Ks,boundType,Bs,chi2_elements)
    if boundType=='relative':
        plot_parameters_relative(pars,TMD,Bs,legendInfo)
    else:
        plot_parameters_absolute(pars,TMD,Bs,legendInfo)
    plot_orbitalContent(pars,data.TMD,legendInfo)
    plot_bands(tb_en,data,legendInfo)
    plt.show()
def plot_bands(tb_en,data,legendInfo):
    """ Plot bands in comparison with ARPES data. """
    fit_data = data.fit_data
    fig = plt.figure(figsize=(15,9))
    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[10, 1],
        hspace=0
    )
    ax = fig.add_subplot(gs[0])
    for b in range(fit_data.shape[1]-3):
        targ = np.argwhere(np.isnan(fit_data[:,3+b]))    #select only non-nan values
        en_pars = copy.copy(tb_en[b,:])
        en_pars[targ] = np.nan
        ax.plot(
            fit_data[:,0],
            fit_data[:,3+b],
            label='ARPES' if b == 0 else '',
            zorder=1,
            color='r',
            marker='o',
            markersize=10,
            mew=1,
            mec='k',
            mfc='firebrick'
        )
        ax.plot(
            fit_data[:,0],
            en_pars,
            ls='-',
            label='Fit' if b == 0 else '',
            zorder=3,
            color='skyblue',
            marker='s',
            markersize=10,
            mew=1,
            mec='k',
            mfc='deepskyblue'
        )
    #
    s_m = 15
    s_ = 20
    s_p = 30
    ks = [fit_data[0,0],4/3*np.pi/cfs.dic_params_a_mono[data.TMD],fit_data[-1,0]]
    ax.set_xticks(ks,[r"$\Gamma$",r"$K$",r"$M$"],size=s_)
    for i in range(len(ks)):
        ax.axvline(ks[i],color='k',lw=0.5)
    ax.set_xlim(ks[0],ks[-1])
    ax.set_ylabel('Energy [eV]',size=s_)
    label_y = []
    if fit_data.shape[1]==9:
        ticks_y = np.linspace(np.max(fit_data[:,3])+0.2,np.min(fit_data[~np.isnan(fit_data[:,6]),6])-0.2,5)
        for i in ticks_y:
            label_y.append("{:.1f}".format(i))
        ax.set_yticks(ticks_y,label_y,size=s_m)
    plt.legend(fontsize=20)
    ax.set_title("Bands comparison",size=s_p)

    ax2 = fig.add_subplot(gs[1])
    addLegendResult(legendInfo,ax2)
    plt.subplots_adjust(
        left = 0.083,
        bottom = 0.045,
        right = 0.893,
        top = 0.95,
        wspace = 0.06,
        hspace = 0.2
    )
def plot_parameters_relative(pars,TMD,Bs,legendInfo):
    """ Plot parameters values and differece wrt DFT parameters. """
    lenP = len(pars)
    #
    DFT_values = cfs.initial_pt[TMD]
    fig = plt.figure(figsize=(19,9))
    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[10, 1],
        hspace=0
    )
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(np.arange(lenP),pars-DFT_values,
            color='r',width=0.4,align='edge',zorder=10)
    ax2 = ax1.twinx()
    ax2.bar(np.arange(lenP),(pars-DFT_values)/abs(np.array(DFT_values))*100,
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
    s_ = 10
    s_p = 15
    ax1.set_ylim(-max(abs(pars-DFT_values)),max(abs(pars-DFT_values)))
    ax1.set_ylabel("Absolute difference from DFT",size=s_p,color='r')
    ax1.set_xticks(np.arange(lenP),cfs.list_formatted_names_all,size=s_)
    ax1.set_xlabel("Parameter name",size=s_p)
    ax1.tick_params(axis='y', labelsize=s_,labelcolor='r')
    ax1.set_xlim(-0.5,lenP-0.5)
    top_ax = ax1.secondary_xaxis('top')
    top_ax.set_xticks(np.arange(lenP),["%d"%i for i in np.arange(lenP)],size=s_)
    top_ax.set_xlabel("Parameter index",size=s_p)
    #ax2
    rp,rpz,rpxy,rl = Bs
    rmax = max([rp,rpz,rpxy,rl])
    ticks_y = np.linspace(-rmax*100,rmax*100,5)
    label_y = ["{:.1f}".format(i)+r"%" for i in ticks_y]
    ax2.set_yticks(ticks_y,label_y,size=s_,color='b')
    ax2.set_ylabel("Relative distance from DFT",size=s_p,color='b')
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

    axl = fig.add_subplot(gs[1])
    addLegendResult(legendInfo,axl)

    fig.tight_layout()
def plot_parameters_absolute(pars,TMD,Bs,legendInfo):
    """ Plot absolute values of barameters with different bounds.
    """
    DFT_pars = cfs.initial_pt[TMD]
    npars = pars.shape[0]

    fig = plt.figure(figsize=(19,9))
    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[10, 1],
        hspace=0
    )
    fig.patch.set_facecolor("#F7F7F7")
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#F7F7F7")

    x     = np.arange(npars)
    # group background bands
    group_colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#64B5CD"]
    group_bounds = [(0,6),(7,27),(28,35),(36,39),(40,40),(41,42)]
    group_labels = [
    "Epsilon", "t_1", "t_5",
    "t_6", "", "SOC",
    ]
    for gi, (start, end) in enumerate(group_bounds):
        ax.axvspan(start - 0.5, end + 0.5, color=group_colors[gi],
                   alpha=0.07, zorder=0)
    param_colors = [""] * npars
    param_bound  = [None] * npars
    has_bound = [0, 1, 2, 3, 5]
    b_idx = 0
    for gi, (start, end) in enumerate(group_bounds):
        for i in range(start, end + 1):
            param_colors[i] = group_colors[gi]
        if gi in has_bound:
            for i in range(start, end + 1):
                param_bound[i] = Bs[b_idx]
            b_idx += 1
    bar_w = 0.8
    for i in range(npars):
        val = pars[i]
        ref = DFT_pars[i]
        c   = param_colors[i]
        # ── bar ──────────────────────────────────────────────────────────
        ax.bar(i, abs(val), width=bar_w, bottom=min(0, val),
               color=c, alpha=0.80, linewidth=0.3, edgecolor="white", zorder=3)
        # ── reference line: bold tick across the full bar width ──────────
        hw = bar_w * 0.48
        #ax.plot([i - hw, i + hw], [ref, ref],
        #        color="white", lw=2.5, zorder=5, solid_capstyle="butt")
        ax.plot([i - hw, i + hw], [ref, ref],
                color="#111",  lw=1.2, zorder=6, solid_capstyle="butt",
                linestyle='-')#(0, (3, 2)))   # short dash so it reads as a marker
        # ── value label (rotated, inside or just outside) ─────────────────
        label = f"{val:+.2f}"
        if abs(val) > 0.20:
            ax.text(i, val / 2, label, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                    rotation=90, zorder=7)
        else:
            yo  = val + (0.035 if val >= 0 else -0.035)
            va_ = "bottom" if val >= 0 else "top"
            ax.text(i, yo, label, ha="center", va=va_,
                    fontsize=8, color="#333", rotation=90, zorder=7)
        # ── bound markers ─────────────────────────────────────────────────
        if param_bound[i] is not None:
            b = param_bound[i]
            for sign in (1, -1):
                ax.plot([i - 1/2, i + 1/2], [sign*b, sign*b],
                        color="#111", lw=1.4, ls="-", zorder=5, alpha=0.75)
    # ── axes ──────────────────────────────────────────────────────────────
    s_ = 12
    s_p = 15
    ax.set_xticks(x)
    ax.set_xticklabels(cfs.list_formatted_names_all, rotation=55, ha="right",
                       fontsize=s_, fontfamily="monospace")
    ax.set_xlim(-0.4, npars - 0.6)
    ax.set_ylabel("Value", fontsize=s_p, labelpad=6)
    ax.axhline(0, color="#555", lw=0.8, zorder=4)
    ax.set_title("Parameter Overview — 43 parameters across 6 groups",
                 fontsize=20, fontweight="bold", pad=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(bottom=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", ls=":", lw=0.5, color="#bbb", zorder=0)
    # group separators + labels
    ylim_top = ax.get_ylim()[1]
    for gi, (start, end) in enumerate(group_bounds[:-1]):
        ax.axvline(end + 0.5, color="#aaa", lw=0.7, zorder=2)
    for gi, (start, end) in enumerate(group_bounds):
        ax.text((start+end)/2, ylim_top*0.97, group_labels[gi],
                ha="center", va="top", fontsize=s_,
                color=group_colors[gi], fontweight="bold", zorder=7)

    axl = fig.add_subplot(gs[1])
    addLegendResult(legendInfo,axl)

    fig.tight_layout()
def plot_orbitalContent(pars,TMD,legendInfo):
    """ Plot orbital content in the BZ cut: G-K-M-G """

    """ BZ cut points """
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
    hopping = cfs.find_t(pars)
    epsilon = cfs.find_e(pars)
    offset = pars[-3]
    #
    HSO = cfs.find_HSO(pars[-2:])
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
    indM = Ngk+Nkm
    """ Plot """
    fig = plt.figure(figsize=(15,9))
    s_m = 15
    s_ = 20
    s_p = 30
    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[10, 1],
        hspace=0
    )
    ax = fig.add_subplot(gs[0])

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
                       fontsize=s_,
                       handletextpad=0.35,
                       handlelength=0.5
                       )
    ax.add_artist(legend)

    ax.set_ylim(-4,3)
    ax.set_xlim(0,Nk-1)

    ax.axvline(Ngk,color='k',lw=1,zorder=-1)
    ax.axvline(Ngk+Nkm,color='k',lw=1,zorder=-1)
    ax.axhline(0,color='k',lw=1,zorder=-1)

    ax.set_xticks([0,Ngk-1,Ngk+Nkm-1,Nk-1],[r"$\Gamma$",r"$K$",r"$M$",r"$\Gamma$"],size=s_)
    ax.set_ylabel("Energy [eV]",size=s_)
    ax.tick_params(axis='y',labelsize=s_m)
    ax.set_title("Orbital content of bands",size=s_p)

    axl = fig.add_subplot(gs[1])
    addLegendResult(legendInfo,axl)
    plt.subplots_adjust(
        left = 0.083,
        bottom = 0.045,
        right = 0.893,
        top = 0.95,
        wspace = 0.06,
        hspace = 0.2
    )
def addLegendResult(legendInfo,ax):
    TMD, Ks, boundType, Bs, chi2_elements = legendInfo
    # Text
    txt = TMD + '\n'
    txt_Bs = ['gen', 'z  ', 'xy ', 'soc'] if boundType=='relative' else ['eps','t_1','t_5','t_6','soc']
    txt += '-'*10 + '\n'
    txt += 'Boundaries: '+boundType+'\n'
    txt += '-'*10 + '\n'
    for i in range(len(Bs)):
        txt += txt_Bs[i] + ': %s'%Bs[i]+'\n'
    txt += '-'*10 + '\n'
    txt += 'Constants\n'
    txt += '-'*10 + '\n'
    for i in range(6):
        txt += 'K%s: %6f'%(i+1,Ks[i])+'\n'
    txt += '-'*10 + '\n'
    txt += 'Function values\n'
    txt += '-'*10 + '\n'
    txt_chiv = ['Chi2 energy bands','K1 pars distance','K2 M orb content','K3 G/K orb content','K4 minimum at K','K5 band gap']
    for i in range(6):
        txt += txt_chiv[i]+':\n    %.6f'%chi2_elements[i]+'\n'
    #
    box_dic = dict(boxstyle='round',facecolor='white',alpha=1)
    ax.text(
        0.0,0.,
        txt,
        bbox = box_dic,
        transform=ax.transAxes,
        fontsize=15,
        fontfamily='monospace',
    )
    ax.axis('off')
def plot_measure(measure, Ks, Bs, global_idx: int, tmd: str, cutoff: float, title: str, fig, ax):
    """ Plot grid of results """
    x_vals, y_vals, grid = build_grid(measure, Ks, Bs)

    # Nan the large values
    grid[grid>cutoff] = np.nan

    # --- global minimum ---
    min_measure = measure[global_idx]
    min_Ks      = Ks[global_idx]
    min_Bs      = Bs[global_idx]

    print(f"Global measure minimum : {min_measure:.6g}  (index {global_idx})")
    print(f"  Ks = {min_Ks}")
    print(f"  Bs = {min_Bs}")

    img = ax.pcolormesh(
        x_vals,
        y_vals,
        grid,
        shading="nearest",
        cmap="viridis_r",
    )
    #ax.set_xscale('log',base=2)
    #ax.set_yscale('log',base=2)

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(r"$\min$ measure", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

    # Mark the global minimum on the grid
    ax.scatter(
        min_Ks[1], min_Ks[2],
        marker="*", s=220, color="red", zorder=5,
        label="global minimum",
    )

    # --- legend with full parameter values ---
    ks_str = "\n".join(f"  K{i+1} = {min_Ks[i]:.6g}" for i in range(len(min_Ks)))
    bs_str = "\n".join(f"  B{i} = {min_Bs[i]:.6g}" for i in range(len(min_Bs)))
    legend_text = (
        f"measure min = {min_measure:.6g}\n"
        f"Ks:\n{ks_str}\n"
        f"Bs:\n{bs_str}"
    )

    ax.scatter([], [], marker="*", color="red", label=legend_text)
    ax.legend(
        fontsize=8,
        loc="upper right",
        framealpha=0.85,
        handlelength=1.2,
        borderpad=0.8,
    )

    ax.set_xlabel(r"$K_2$", fontsize=13)
    ax.set_ylabel(r"$K_3$", fontsize=13)
    ax.set_title(f"{tmd} : measure = {title}",
        fontsize=13,
    )
def build_grid(measure, Ks, Bs):
    """ Build 2D grid:
    x = Ks[:,1],
    y = Ks[:,2],
    z = min(measure) over the rest
    """
    x_vals = np.unique(Ks[:, 1])
    y_vals = np.unique(Ks[:, 2])

    grid = np.full((len(y_vals), len(x_vals)), np.nan)

    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    for i in range(len(measure)):
        xi = x_idx[Ks[i, 1]]
        yi = y_idx[Ks[i, 2]]
        if np.isnan(grid[yi, xi]) or measure[i] < grid[yi, xi]:
            grid[yi, xi] = measure[i]

    return x_vals, y_vals, grid

""" Bounds and other functions """
def get_bounds(in_pt,args_minimization):
    if args_minimization['boundType']=='relative':
        rp, rpz, rpxy, rl = args_minimization['Bs']
        Bounds = []
        for i in range(in_pt.shape[0]):     #tb parameters
            if i in indOff: #offset
                temp = (-3,0)
            elif i in indSOC: #SOC
                r = rl*abs(in_pt[i])
            elif i in indPz:
                r = rpz*abs(in_pt[i])
            elif i in indPxy:
                r = rpxy*abs(in_pt[i])
            else:
                r = rp*abs(in_pt[i])
            temp = (in_pt[i]-r,in_pt[i]+r)
            Bounds.append(temp)
    elif args_minimization['boundType']=='absolute':
        peps, pt1, pt5, pt6, pl = args_minimization['Bs']
        Bounds = []
        for i in range(in_pt.shape[0]):     #tb parameters
            if i in indOff: #offset
                temp = (-3,0)
            elif i in indSOC: #SOC
                temp = (-pl, pl)
            elif i in indEps:
                temp = (-peps, peps)
            elif i in indT1:
                temp = (-pt1, pt1)
            elif i in indT5:
                temp = (-pt5, pt5)
            elif i in indT6:
                temp = (-pt6, pt6)
            else:
                raise ValueError("Index not in any list for bounds: "+str(i))
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
    distance = np.sum(np.absolute((pars[:-3]-DFT_values[:-3])/DFT_values[:-3])) + np.sum(np.absolute((pars[-2:]-DFT_values[-2:])/DFT_values[-2:]))
    distance /= (pars.shape[0]-1)
    return distance

def get_home_dn(machine):
    """ Define the name of the directory where I'm working in order to save the data. """
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/Code/monolayer_v3.0/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/monolayer_v3.0/'
    elif machine == 'maf':
        return '/users/rossid/monolayer_v3.0/'



































