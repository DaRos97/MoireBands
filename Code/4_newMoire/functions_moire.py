import numpy as np
import scipy
import CORE_functions as cfs
import itertools
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.special import wofz      #for fitting weights in EDC
from lmfit import Model      #for fitting weights in EDC

machine = cfs.get_machine(os.getcwd())

def get_pars(ind):
    lMonolayer_type = ['fit',]
    lSample = ['S3',]  #this and theta are related! Sample needed also for interlayer parameters' choice
    lStacking = ['P',]
    lW2p = [0.0]#np.linspace(-0.05,0.05,5)
    lW2d = [0.0]#np.linspace(-0.05,0.05,5)
    pars_Vgs = [0.0,]        #Moire potential at Gamma
    pars_Vks = [0.007,]             #Moire potential at K
    phi_G = [np.pi/3,]                #Phase at Gamma
    phi_K = [-106*np.pi/180,]       #Phase at K
    lnShells = [0,]                       #number of BZ circles
    lCuts = ['Kp-G-K',]#'Kp-G-K',]          #'Kp-G-K-Kp'
    lKpts = [100,]
    #
    ll = [lMonolayer_type,lStacking,lW2p,lW2d,pars_Vgs,pars_Vks,phi_G,phi_K,lSample,lnShells,lCuts,lKpts,]
    return list(itertools.product(*ll))[ind]

def import_monolayer_parameters(monolayer_type,machine):
    """Import monolayer parameters, either DFT or fit ones."""
    hopping = {}
    epsilon = {}
    HSO = {}
    offset = {}
    for TMD in cfs.TMDs:
        temp = np.load(get_inputs_dn(machine)+'result_'+TMD+'.npy') if monolayer_type=='fit' else np.array(cfs.initial_pt[TMD])
        hopping[TMD] = cfs.find_t(temp)
        epsilon[TMD] = cfs.find_e(temp)
        HSO[TMD] = cfs.find_HSO(temp[-2:])
        offset[TMD] = temp[-3]
    return (hopping,epsilon,HSO,offset)

def H_moire(pars_V):
    """
    Compute moire potential matrix coupling two mini-BZ.
    It is a 22x22 matrix with different values of moirè potential for in and out of plane orbitals.
    The phase needs to be chosen cottectly in order to be consistent with the interlayer coupling.
    """
    V_G,V_K,psi_G,psi_K = pars_V
    Id = np.zeros((22,22),dtype = complex)
    out_of_plane = V_G*np.exp(1j*psi_G)
    in_plane = V_K*np.exp(1j*psi_K)
    list_out = (0,1,2,5,8)      #out-of-plane orbitals (all ones containing a z)
    list_in = (3,4,6,7,9,10)    #in-plane orbitals 
    for i in list_out:
        Id[i,i] = out_of_plane
        Id[i+11,i+11] = out_of_plane
    for i in list_in:
        Id[i,i] = in_plane
        Id[i+11,i+11] = in_plane
    return Id

def diagonalize_matrix(*args):
    """
    Compute and diagonalize big matrix and save to file the result.
    """
    nShells, nCells, K_list, monolayer_type, parsInterlayer, theta, pars_V, energy_fn, save_data_energy, disp = args
    kPts = K_list.shape[0]
    #Monolayer parameters
    pars_monolayer = import_monolayer_parameters(monolayer_type,machine)
    #Moire parameters
    G_M = cfs.get_reciprocal_moire(theta/180*np.pi)     #7 reciprocal moire lattice vectors
    moireHam = H_moire(pars_V)
    pars_moire = (nCells,G_M,moireHam)
    #
    evals = np.zeros((kPts,nCells*44))
    evecs = np.zeros((kPts,nCells*44,nCells*44),dtype=complex)
    look_up = lu_table(nShells)
    if disp:
        from tqdm import tqdm
    else:
        tqdm = cfs.tqdm
    for i in tqdm(range(kPts),desc="Diagonalization of Hamiltonian"):
        K_i = K_list[i]
        H_tot = big_H(K_i,nShells,look_up,pars_monolayer,parsInterlayer,pars_moire)
        evals[i],evecs[i] = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
    if save_data_energy:
        np.savez(energy_fn,evals=evals,evecs=evecs)
    return evals, evecs

def lu_table(nShells):
    """
    Computes the look-up table for the index of the mini-BZ in terms of the reciprocal lattice vector indexes.
    """
    n_cells = int(1+3*nShells*(nShells+1))
    lu = []
    for n in range(0,nShells+1):      #circles go from 0 (central BZ) to N included
        i = 0
        j = 0
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       #mini-BZ index
            if s == np.sign(n)*(1+(n-1)*n*3):
                lu.append((n,0))
            else:
                lu.append((lu[-1][0]+cfs.m_list[i][0],lu[-1][1]+cfs.m_list[i][1]))
                if j == n-1:
                    i += 1
                    j = 0
                else:
                    j += 1
    return lu

def big_H(K_,nShells,lu,pars_monolayer,parsInterlayer,pars_moire):
    """
    Compute the large Hamiltonian containing all the moire replicas.
    Each k-point has 44*nCells dimension, with nCells the number of mini-BZ considered.
    The basis is:
        0 to 22*nCells-1 -> WSe2, 22*nCells to 44*nCells-1 -> WS2.
        0 to 21 has the basis of the monolayer for mini BZ #0, and so on for the other mBZs.
    """
    hopping,epsilon,HSO,offset = pars_monolayer
    nCells,G_M,moireHam = pars_moire
    #
    Ham = np.zeros((nCells*44,nCells*44),dtype=complex)
    # Dispersion args
    args_WSe2 = (hopping['WSe2'],epsilon['WSe2'],HSO['WSe2'],cfs.dic_params_a_mono['WSe2'],offset['WSe2'])
    args_WS2 = (hopping['WS2'],epsilon['WS2'],HSO['WS2'],cfs.dic_params_a_mono['WS2'],offset['WS2'])
    # Momentum -> needed for interlayer
    Kns = np.zeros((nCells,2))
    for i in range(nCells):
        Kns[i] = K_ + G_M[1]*lu[i][0] + G_M[2]*lu[i][1]
    #
    orbpList = [8,19]       #list of indexes of p-orbitals for interlayer coupling -> p_z^e
    orbdList = [5,16]       #list of indexes of d-orbitals for interlayer coupling -> d_z^2
    psiInterlayer = 0 if parsInterlayer['stacking']=='P' else 2*np.pi/3
    # Monolayer dispersions and interlayer coupling
    for n in range(nCells):
        Kn = Kns[n]
        Ham [n*22:(n+1)*22,n*22:(n+1)*22] = cfs.H_monolayer(Kn,*args_WSe2)
        Ham [(nCells+n)*22:(nCells+n+1)*22,(nCells+n)*22:(nCells+n+1)*22] = cfs.H_monolayer(Kn,*args_WS2)
        # w1 p and d: e-e, e-o, o-e and o-o (last two with minus sign)
        for iSO in [0,11]:
            Ham[n*22 + 8 + iSO,(nCells+n)*22 + 8 + iSO] += parsInterlayer['w1p']
#        for orbp in orbpList:
#            Ham[n*22 + orbp,(nCells+n)*22 + orbp] += parsInterlayer['w1p']
        for orbd in orbdList:
            Ham[n*22 + orbd,(nCells+n)*22 + orbd] += parsInterlayer['w1d']
        if 0:# w2 p and d
            for ng in [1,2,3,4,5,6]:      #index of G_M to consider in the interlayer
                diff = np.linalg.norm(np.absolute(Kn - Kns + G_M[ng]),axis=1)
                inds = np.where(diff<1e-10)[0]
                for ind in inds:
                    for iSO in [0,11]:
                        Ham[n*22 + 2 + iSO,(nCells+ind)*22 + 2 + iSO] += parsInterlayer['w2p']/2
                        Ham[n*22 + 2 + iSO,(nCells+ind)*22 + 8 + iSO] += parsInterlayer['w2p']/2
                        Ham[n*22 + 8 + iSO,(nCells+ind)*22 + 2 + iSO] -= parsInterlayer['w2p']/2
                        Ham[n*22 + 8 + iSO,(nCells+ind)*22 + 8 + iSO] -= parsInterlayer['w2p']/2
#                    for orbp in orbpList:
#                        Ham[n*22 + orbp,(nCells+ind)*22 + orbp] += parsInterlayer['w2p']
                    for orbd in orbdList:
                        Ham[n*22 + orbd,(nCells+ind)*22 + orbd] += parsInterlayer['w2d']*np.exp(-1j*(-1)**(ng-1)*psiInterlayer)
    Ham[nCells*22:,:nCells*22] = np.copy(Ham[:nCells*22,nCells*22:].T.conj())
    #Moirè replicas
    for n in range(0,nShells+1):      #Circles
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       #Indices inside the circle
            ind_s = lu[s]
            for i in cfs.m_list:
                ind_nn = (ind_s[0]+i[0],ind_s[1]+i[1])  #nn-> nearest neighbour
                try:
                    nn = lu.index(ind_nn)
                except:
                    continue
                g = cfs.m_list.index(i)
                Vup = moireHam if g%2 else moireHam.conj()
                Ham [s*22:(s+1)*22,nn*22:(nn+1)*22] += Vup
                Ham [nCells*22 + s*22:nCells*22 +(s+1)*22, nCells*22 + nn*22:nCells*22 +(nn+1)*22] += Vup#.conj()
    return Ham

def weight_spreading(weight,K_temp,E_temp,K_list,e_grid,pars_spread):
    """Compute the weight spreading in k and e.

    Parameters
    ----------
    weight : float
        Weight to spread.
    K : float
        Momentum position of weight.
    E : float
        Energy position of weight.
    k_grid : np.ndarray
        Grid of values over which evaluate the spreading in momentum.
    e_grid : np.ndarray
        Grid of values over which evaluate the spreading in energy.
    pars_spread : tuple
        Parameters of spreading: gamma_k, gamma_e, type_of_spread (Gauss or Lorentz).

    Returns
    -------
    np.ndarray
        Grid of energy and momentum values over which the weight located at K,E has been spread using the type_of_spread function by values spread_K and spread_E.
    """
    spread_k,spread_E,type_of_spread = pars_spread
    k_grid = np.linalg.norm(K_list-K_temp,axis=1)[:,None]
    if type_of_spread == 'Lorentz':
        E2 = spread_E**2
        K2 = spread_k**2
        return spread_E/np.pi*spread_k/np.pi*weight/(k_grid**2+K2)/((e_grid-E_temp)**2+E2)
    elif type_of_spread == 'Gauss':
        return weight*np.exp(-(k_grid/spread_k)**2)*np.exp(-((e_grid-E_temp)/spread_E)**2)

def plot_rk(theta,kList,cut,save_plot_rk):
    """
    Plot real and momentum space lattice with specific twist.
    Plot also momentum space cut and zoom over mini-BZ.
    """
    kPts = kList.shape[0]
    G_M = cfs.get_reciprocal_moire(theta/180*np.pi)     #7 reciprocal moire lattice vectors
    As_WSe2, Bs_WSe2 = cfs.get_lattice_vectors('WSe2')
    As_WS2,  Bs_WS2  = cfs.get_lattice_vectors('WS2',theta)
    c1,c2 = ('r','b')
    s_ = 20
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(121)
    ax.axis('off')
    ax.set_aspect('equal')
    for i in range(6):
        ax.arrow(0,0,As_WSe2[i][0],As_WSe2[i][1],color=c1,head_width=0.1,length_includes_head=True)
        ax.arrow(0,0,As_WS2[i][0],As_WS2[i][1],color=c2,head_width=0.1,length_includes_head=True)
    pt0_WSe2 = (As_WSe2[0]+As_WSe2[1])/3
    pt0_WS2 = (As_WS2[0]+As_WS2[1])/3
    for i in range(6):
        pta = cfs.R_z(np.pi/3*i) @ pt0_WSe2
        ptb = cfs.R_z(np.pi/3*(i+1)) @ pt0_WSe2
        ax.plot([pta[0],ptb[0]],[pta[1],ptb[1]],color=c1,marker='o')
        pta = cfs.R_z(np.pi/3*i) @ pt0_WS2
        ptb = cfs.R_z(np.pi/3*(i+1)) @ pt0_WS2
        ax.plot([pta[0],ptb[0]],[pta[1],ptb[1]],color=c2,marker='*')
    ax.set_title("Real space WSe2 (red) and WS2 (blue), with rotation "+"{:.1f}".format(theta)+"°",fontsize=s_)
    #
    ax = fig.add_subplot(122)
    ax.set_aspect('equal')
    ax.axis('off')
    c1,c2 = ('orangered','dodgerblue')
    sizes = (1 + np.arange(kPts)/kPts)**5
    ax.scatter(kList[:,0],kList[:,1],color='k',s=sizes)
    for i in range(6):
        ax.arrow(0,0,Bs_WSe2[i][0],Bs_WSe2[i][1],color=c1,head_width=0.1,length_includes_head=True)
        ax.arrow(0,0,Bs_WS2[i][0],Bs_WS2[i][1],color=c2,head_width=0.1,length_includes_head=True)
    pt0_WSe2 = (Bs_WSe2[0]+Bs_WSe2[1])/3
    pt0_WS2 = (Bs_WS2[0]+Bs_WS2[1])/3
    for i in range(6):
        pta = cfs.R_z(np.pi/3*i) @ pt0_WSe2
        ptb = cfs.R_z(np.pi/3*(i+1)) @ pt0_WSe2
        ax.plot([pta[0],ptb[0]],[pta[1],ptb[1]],color=c1,marker='o')
        pta = cfs.R_z(np.pi/3*i) @ pt0_WS2
        ptb = cfs.R_z(np.pi/3*(i+1)) @ pt0_WS2
        ax.plot([pta[0],ptb[0]],[pta[1],ptb[1]],color=c2,marker='*')
    # Moire part
    c3,c4 = ('g','limegreen')
    pt0 = pt0_WSe2 + (G_M[2]+G_M[3])/3
    for i in range(1,7):    #moire vectors
        ax.arrow(pt0[0],pt0[1],G_M[i][0],G_M[i][1],head_width=0.01,length_includes_head=True,color='cyan' if i==1 else c3)
    pt0_m = (G_M[1]+G_M[2])/3
    for i in range(6):      #moire mini-BZ
        pta = pt0 + cfs.R_z(np.pi/3*i) @ pt0_m
        ptb = pt0 + cfs.R_z(np.pi/3*(i+1)) @ pt0_m
        ax.plot([pta[0],ptb[0]],[pta[1],ptb[1]],color=c4,marker='o',markersize=1)
    axins = zoomed_inset_axes(ax, zoom=7, loc=2)
    for i in range(6):
        pta = cfs.R_z(np.pi/3*i) @ pt0_WSe2
        ptb = cfs.R_z(np.pi/3*(i+1)) @ pt0_WSe2
        axins.plot([pta[0],ptb[0]],[pta[1],ptb[1]],color=c1,marker='o')
        pta = cfs.R_z(np.pi/3*i) @ pt0_WS2
        ptb = cfs.R_z(np.pi/3*(i+1)) @ pt0_WS2
        axins.plot([pta[0],ptb[0]],[pta[1],ptb[1]],color=c2,marker='*')
    for i in range(1,7):    #moire vectors
        axins.arrow(pt0[0],pt0[1],G_M[i][0],G_M[i][1],head_width=0.01,length_includes_head=True,color='cyan' if i==1 else c3)
    pt0_m = (G_M[1]+G_M[2])/3
    for i in range(6):      #moire mini-BZ
        pta = pt0 + cfs.R_z(np.pi/3*i) @ pt0_m
        ptb = pt0 + cfs.R_z(np.pi/3*(i+1)) @ pt0_m
        axins.plot([pta[0],ptb[0]],[pta[1],ptb[1]],color=c4,marker='o',markersize=1)
    delta = 1.*np.linalg.norm(G_M[1])
    axins.set_xlim(pt0[0] - delta, pt0[0] + delta)
    axins.set_ylim(pt0[1] - delta, pt0[1] + delta)
    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    ax.set_title("Momentum space cut: "+cut+"",fontsize=s_)

    if save_plot_rk:
        print("Saving figure real and momentum space")
        figname = get_figures_dn(machine) + 'rk_' + "{:.1f}".format(theta) + "_" + cut + '.png'
        fig.savefig(figname)

    plt.show()

def voigt(x, amplitude, center, sigma, gamma):
    """
    Voigt profile: convolution of Lorentzian and Gaussian
    - sigma: Gaussian standard deviation
    - gamma: Lorentzian half-width at half-maximum
    """
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def double_voigt(x, amp1, cen1, sig1, gam1, amp2, cen2, sig2, gam2):
    return (voigt(x, amp1, cen1, sig1, gam1) +
            voigt(x, amp2, cen2, sig2, gam2))

import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

def EDC(args,spreadE=0.03,disp=False,plot=False,figname=''):
    """
    Compute energy distance of side bands crossing from main band.
    We do it by diagonalizing the Hamiltonian at the desired k point.
    We evaluate and extract the weights BELOW the main band (carefull to the SOC degeneracy at Gamma).
    We spread the weights with a Lorentzian of width 30 meV.
    We fit the intensity profile with 2 Lorentzians (convoluted with a Gaussian).
    """
    nCells = args[1]
    # Evals, evecs and weights at Gamma
    e_, ev_ = diagonalize_matrix(*args)
    evals = e_[0]
    evecs = ev_[0]
    ab = np.absolute(evecs)**2
    weights = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    # Extract energies and weights we need -> TVB: main band and all side bands, just on TOP layer
    indexMainBand = 28*nCells - 1
    energyMainBand = evals[indexMainBand]
    weightMainBand = weights[indexMainBand]
    fullEnergyValues = evals  [indexMainBand-nCells*2+1:indexMainBand+1]
    fullWeightValues = weights[indexMainBand-nCells*2+1:indexMainBand+1]
    # Define finer energy list for weight spreading: slightly larger for better spreading shape
    energyList = np.linspace(-1.0,-0.4,200)      #we chose this from experimental data
    weightList = np.zeros(len(energyList))
    if energyMainBand > energyList[-1] and 1:     # Check we are in right window
        if disp:
            print("TVB energy higher than -0.4 eV -> clearly wrong")
        else:
            return -1
    if np.max(fullWeightValues[:-2]) > weightMainBand and 1:     # Check the main band has higher weight
        if disp:
            print("TVB weight is not the maximum -> clearly wrong")
        else:
            return -1
    for i in range(len(fullEnergyValues)):
        weightList += spreadE/np.pi * fullWeightValues[i] / ((energyList-fullEnergyValues[i])**2+spreadE**2)
    try:    # Fit the spreaded weights with two Lorentzian peaks convoluted with a Gaussian
        model = Model(double_voigt, independent_vars=['x'])
        # Initial guess and boundaries
        params = model.make_params(amp1=1.57, cen1=-0.67, sig1=0.005, gam1=0.03,
                               amp2=0.41, cen2=-0.76, sig2=0.005, gam2=0.03)
        params['amp1'].set(min=1,max=10)
#        params['amp1'].set(min=0.1,max=10)
        params['sig1'].set(min=0,max=1)
        params['cen1'].set(min=-0.73,max=-0.6)
        params['gam1'].set(min=1e-5,max=0.08)
        params['amp2'].set(min=0.1,max=5)
#        params['amp2'].set(min=0.1,max=10)
        params['sig2'].set(min=0,max=1)
        params['cen2'].set(min=-0.81,max=-0.73)
        params['gam2'].set(min=1e-5,max=0.08)
        result = model.fit(weightList, params, x=energyList)
        # Checks on the result
        if result.chisqr > 100:     # Bad fit
            if disp:
                print("Chisqr high")
            raise ValueError
        if 1:
            for name, param in result.params.items():   #check if parameetrs are hitting the boundaries I set
                if (param.min is not None and abs(param.value - param.min) < 1e-10) or (param.max is not None and abs(param.value - param.max) < 1e-10):
                    if disp:
                        print("Par "+name+" at boundary: ",param)
#                    break
                    raise ValueError
        distance = result.best_values['cen1'] - result.best_values['cen2']
        fitSuccess = True
        if disp:
            print(result.fit_report())
    except Exception as e:
        if disp:
            print("Fit didn't work, exception ", e)
        distance = -1
        fitSuccess = False
    if plot:
        imageAroundGamma = 1
        if imageAroundGamma:       #Compute image zoom around Gamma
            fig = plt.figure(figsize=(20,13))
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.5])
            # Need to compute some points around Gamma
            kPts = 101 #has to be odd
            range_k = 0.6
            kList = np.zeros((kPts,2))
            kList[:,0] = np.linspace(-range_k,range_k,kPts)
            nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, moir_pars, _, __, disp = args
            args2 = (nShells, nCells, kList, monolayer_type, parsInterlayer, theta, moir_pars, '', False, True)
            evals2, evecs2 = diagonalize_matrix(*args2)
            weights2 = np.zeros((kPts,nCells*44))
            for i in range(kPts):
                ab2 = np.absolute(evecs2[i])**2
                weights2[i,:] = np.sum(ab2[:22,:],axis=0) + np.sum(ab2[22*nCells:22*(1+nCells),:],axis=0)
            #Figure of points
            ax = fig.add_subplot(gs[0,0])
            kLine = kList[:,0]
            for n in range(16*nCells,28*nCells):
                ax.plot(kLine,evals2[:,n],color='r',lw=0.3,zorder=1)
                ax.scatter(kLine,evals2[:,n], s=weights2[:,n]*100,
                           color='b',lw=0,zorder=3)
            ax.set_ylim(-1.2,-0.4)
            ax.set_xlim(-range_k,range_k)
            ax.set_ylabel("eV")
            V = args[6][0]
            phiG = args[6][2]/np.pi*180
            w1p = args[4]['w1p']
            w1d = args[4]['w1d']
            stacking = args[4]['stacking']
            ax.text(0.1,0.9,r"stacking: %s, V=%.4f eV, $\varphi$=%.1f°, $w_1^p$=%.4f eV, $w_1^d$=%.4f eV"%(stacking,V,phiG,w1p,w1d),size=20,transform=ax.transAxes)
            #Figure of spread
            ax = fig.add_subplot(gs[1,0])
            E_list = np.linspace(-1.2,-0.4,150)
            spread = np.zeros((kPts,len(E_list)))
            pars_spread = (0.001,0.03,'Lorentz')
            for i in range(kPts):
                for n in range(indexMainBand-nCells*2+1,indexMainBand+1):
                    spread += weight_spreading(weights2[i,n],kList[i],evals2[i,n],kList,E_list[None,:],pars_spread)
            spread /= np.max(spread)        #0 to 1
            map_ = 'gray_r'
            ax.imshow(spread.T[::-1,:]**0.5,
              cmap=map_,
              aspect=kPts/len(E_list)*0.45,     #to coincide by eye between the two images
              interpolation='none'
             )
            ax.set_xticks([])
            ax.set_xlabel('K')
            #
            ax = fig.add_subplot(gs[1,1])
            ax.plot(E_list,spread[kPts//2,:])
            ax.set_xlim(-1,-0.4)
            ax = fig.add_subplot(gs[0,1])
        else:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot()
        # Plot weight spreading
        ax.scatter(energyList,weightList,color='b')
        ax.scatter(fullEnergyValues,fullWeightValues,color='r')
        ax.set_xlim(-1,-0.4)
        if fitSuccess:
            ax.plot(energyList,result.best_fit,color='g',ls='--',lw=2)
            ax.axvline(result.best_values['cen1'],color='r')
            ax.axvline(result.best_values['cen2'],color='r')
        fig.tight_layout()
        if not figname=='':
            fig.savefig(figname)
        plt.show()
        return distance
    return distance


""" FILENAME FUNCTIONS """
def get_fn(*args):
    """
    Get filename for set of parameters.
    """
    fn = ''
    for i,a in enumerate(args):
        t = type(a)
        if t in [str,np.str_]:
            fn += a
        elif t in [int, np.int64]:
            fn += str(a)
        elif t in [float, np.float32, np.float64]:
            fn += "{:.4f}".format(a)
        elif t==dict:
            args = np.copy(list(a.values()))
            fn += get_fn(*args)
        else:
            print("Parameter ",a)
            print("is unsupported type: ",t)
            exit()
        if not i==len(args)-1:
            fn +='_'
    return fn

def get_figures_dn(machine):
    return get_home_dn(machine)+'Figures/'

def get_inputs_dn(machine):
    return get_home_dn(machine)+'Inputs/'

def get_data_dn(machine):
    return get_home_dn(machine)+'Data/'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/Code/4_newMoire/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/4_newMoire/'
    elif machine == 'maf':
        return '/users/rossid/4_newMoire/'

""" LDOS FUNCTIONS """

def LDOS_kList(kPts,G_M):
    """
    Here we have to define the grid of momentum points to sum over.
    Here now is a grid of the mini-BZ.
    """
    G1,G2 = G_M[1:3]
    kList = np.zeros((kPts,kPts,2))
    for ix in range(kPts):
        for iy in range(kPts):
            kList[ix,iy] = G1*ix/kPts + G2*iy/kPts
    kFlat = kList.reshape(-1,2)
    return kFlat

def LDOS_rList(rPts,a_M):
    """
    Here we have to define the real space points to compute.
    """
    a1,a2 = a_M
    rList = np.zeros((rPts,2))
    for i in range(rPts):
        rList[i] = (a1+a2)*i/rPts
    return rList

def get_rs_moire(G_M):
    """
    Here we compute the moirè real-space vectors from the reciprocal ones.
    """
    a_M = 2*np.pi*np.linalg.inv(G_M[1:3])
    a1 = a_M[:,0]
    a2 = cfs.R_z(np.pi/3) @ a1
    return [a1,a2]







