import numpy as np
from itertools import product,islice
import scipy
import CORE_functions as cfs
import os
import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.special import wofz      #for fitting weights in EDC
import lmfit      #for fitting weights in EDC

""" Parameters """
def get_parameters(chunk_id,BZpoint,n_chunks=128):
    """ Get chunks of parameters to compute. """
    if BZpoint=='G':
        listVg = np.linspace(0.005,0.025,21)     # Considered values of moirè potential -> every 1 meV
        listPhi = np.linspace(160,180,21,endpoint=True) /180*np.pi
        listW1p = np.linspace(-2.000,-1.500,51)         # every 5 meV
        listW1d = np.linspace( 0.800, 1.200,41)           # every 5 meV
        filename = cfs.getFilename(
            (
            listVg[0],listVg[-1],len(listVg),int(listPhi[0]/np.pi*180),int(listPhi[-1]/np.pi*180),len(listPhi),
            listW1p[0],listW1p[-1],len(listW1p),listW1d[0],listW1d[-1],len(listW1d)
            )
        )
        #
        grid = product(listVg, listPhi, listW1p, listW1d)
        total_jobs = len(listVg)*len(listPhi)*len(listW1p)*len(listW1d)
    elif BZpoint=='K':
        listVk = np.linspace(0.001,0.020,20)     # Considered values of moirè potential -> every 1 meV
        listPhiK = np.linspace(0,359,360) /180*np.pi
        filename = cfs.getFilename(
            ( listVk[0],listVk[-1],len(listVk),int(listPhiK[0]/np.pi*180),int(listPhiK[-1]/np.pi*180),len(listPhiK) )
        )
        #
        grid = product(listVk, listPhiK)
        total_jobs = len(listVk)*len(listPhiK)
    chunk_size = total_jobs // n_chunks
    remainder = total_jobs % n_chunks
    start = chunk_id * chunk_size + min(chunk_id, remainder)
    end = start + chunk_size + (1 if chunk_id < remainder else 0)
    chunk_iter = islice(grid, start, end)
    print("Total jobs: %d"%total_jobs)
    print("This chunk: %d"%(end-start))
    return chunk_iter,filename

""" EDC """
def EDC(args_diag,sample,BZpoint='G',spreadE=0.03,disp=False,plot=False,machine='loc',showFit=False):
    """
    Compute peak positions of bands by fitting the intensity profile.
    We do it by diagonalizing the Hamiltonian at the desired k point.
    We spread the weights with a Lorentzian of width `spreadE`.
    We fit the intensity profile with 2 Lorentzians (convoluted with a Gaussian).
    Two different fittings (same procedure) for TVB and LVB (lower valence band -> WS2).

    Returns
    -------
    tuple: three positions of fit
    bool: success flag of procedure
    """
    if 0:
        plotBands(args_diag,sample,BZpoint)
    nCells = args_diag[1]
    # Evals, evecs and weights at edcPoint
    e_, ev_ = diagonalize_matrix(*args_diag,machine=machine)
    evals = e_[0]
    evecs = ev_[0]
    ab = np.absolute(evecs)**2
    weights = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    # Bands fitting
    pTVB,successTVB = fitBands('TVB',evals,weights,nCells,spreadE,sample,BZpoint,showFit=showFit)
    if successTVB:
        if BZpoint=='K':
            return pTVB, True
        pLVB,successLVB = fitBands('LVB',evals,weights,nCells,spreadE,sample,BZpoint,showFit=showFit)
        if successLVB:
            return (pTVB[0],pTVB[1],pLVB[0]), True
    return np.nan, False
def fitBands(bandType,evals,weights,nCells,spreadE,sample,BZpoint,showFit=False):
    """
    Uses lmfit to try to fit the intensity profile to 2 Lorentzians convoluted with a Gaussian.
    Used to extract the peak positions of main band and side band crossings.

    Returns
    -------
    tuple: two positions of the peaks
    bool: fitting success flag
    """
    indexB = 28*nCells - 1 if bandType=='TVB' else 26*nCells - 1
    indexL = indexB-2*nCells+1 if BZpoint=='G' else indexB-nCells+1 + np.argmax(weights[indexB-nCells+1:indexB]) +1
    nSOC = 2 if BZpoint=='G' else 1
    energyB = evals[indexB]
    weightB = weights[indexB]
    #
    fullEnergyValues = evals  [indexL:indexB+1]
    fullWeightValues = weights[indexL:indexB+1]
    # Define finer energy list for weight spreading: slightly larger for better spreading shape
    minE, maxE = (-1,-0.5) if bandType=='TVB' else (-1.6,-1.1)
    nE = 251
    if BZpoint=='K':
        minE, maxE, nE = (-0.8,-0.2,301)
    energyList = np.linspace(minE,maxE,nE)      #we chose this from experimental data
    weightList = np.zeros(len(energyList))
    if np.max(fullWeightValues[:-nSOC]) > weightB:     # Check the main band has highest weight
        return (0,0),False
    # Lorentzian spreading
    for i in range(len(fullEnergyValues)):
        weightList += spreadE/np.pi * fullWeightValues[i] / ((energyList-fullEnergyValues[i])**2+spreadE**2)
    # Fit the spreaded weights with two Lorentzian peaks convoluted with a Gaussian
    model = lmfit.Model(two_lorentzian_one_gaussian)
    cen1 = energyList[np.argmax(weightList)]
    cen2 = cen1-0.05 if BZpoint=='G' else cen1-0.15
    params = model.make_params(
        amp1=1.57, cen1=cen1, gam1=0.03,
        amp2=0.41, cen2=cen2, gam2=0.03,
        sig=0.07,
    )
    params['sig'].set(min=1e-6, max=50)        # Gaussian width
    params['gam1'].set(min=1e-6, max=50)       # Lorentzian widths
    params['gam2'].set(min=1e-6, max=50)
    params['amp1'].set(min=0)
    params['amp2'].set(min=0)
    result = model.fit(weightList, params, x=energyList)

    amp1, amp2 = result.best_values['amp1'], result.best_values['amp2']
    cen1, cen2 = result.best_values['cen1'], result.best_values['cen2']#) if amp1>amp2 else (result.best_values['cen2'], result.best_values['cen1'])

    if showFit:   # Plot fit to check
        plotFitResult(energyList,weightList,fullEnergyValues,fullWeightValues,result,sample,bandType,BZpoint)

    if result.success and amp1>1e-3 and amp2>1e-3 and result.redchi<1e-2 and amp1>amp2:
        return (cen1, cen2), True
    else:
        return (0, 0), False
def voigt(x, center, amplitude, gamma, sigma):
    """Single Voigt peak."""
    z = ((x - center) + 1j*gamma) / (sigma*np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))
def two_lorentzian_one_gaussian(x, amp1,cen1,gam1, amp2,cen2,gam2, sig):
    """
    Sum of two Lorentzians, both convolved with the SAME Gaussian.
    Equivalent to two Voigt peaks with shared sigma.
    """

    peak1 = voigt(x, cen1, amp1, gam1, sig)
    peak2 = voigt(x, cen2, amp2, gam2, sig)

    return peak1 + peak2
def plotFitResult(energyList,weightList,fullEnergyValues,fullWeightValues,result,sample,bandType,BZpoint):
    imageBands = False
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    if BZpoint=='G':
        p1,p2,p3 = tuple(cfs.dic_params_edcG_positions[sample] - cfs.dic_params_offset[sample])
    else:
        p1,p2 = tuple(cfs.dic_params_edcK_positions[sample] - cfs.dic_params_offset[sample])
    # Plot weight spreading
    ax.scatter(energyList,weightList,color='b')
    ax.scatter(fullEnergyValues,fullWeightValues,color='r')
    #ax.set_xlim(energyList[0],energyList[-1])
    if result.success:
        ax.plot(energyList,result.best_fit,color='g',ls='--',lw=2)
        ax.axvline(result.best_values['cen1'],color='r',label="fit: cen1=%.4f"%result.best_values['cen1'])
        ax.axvline(result.best_values['cen2'],color='orange',label="fit: cen2=%.4f"%result.best_values['cen2'])
    ax.axhline(0,lw=0.5,zorder=-10,color='k')
    ax.axvline(p1,color='green',lw=2,zorder=-1,label="ARPES: cen TVB=%.4f"%p1)
    ax.axvline(p2,color='lime',lw=2,zorder=-1,label="ARPES: cen EDC=%.4f"%p2)
    if BZpoint=='G':
        ax.axvline(p3,color='blue',lw=2,zorder=-1,label="ARPES: cen LVB=%.4f"%p3)
    ax.set_title(bandType,size=30)
    fig.tight_layout()
    ax.legend(fontsize=15)
    plt.show()
def plotBands(args_diag,sample,BZpoint):
    """ Plot bands around the BZ point """
    args = list(args_diag)
    pts = 151
    kList = np.zeros((pts,2))
    kList[:,0] = np.linspace(-1.5,1.5,pts)
    if BZpoint=='K':
        kList[:,0] += 4*np.pi/3/cfs.dic_params_a_mono['WSe2']
    args[2] = kList
    e_, ev_ = diagonalize_matrix(*args,machine='loc')
    ab = np.absolute(ev_)**2
    #weights = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    nCells = args[1]
    nTVB = 28*nCells
    nBands = 4
    weights = np.zeros(e_.shape)
    for i in range(pts):
        ab = np.absolute(ev_[i])**2
        weights[i,:] = np.sum(ab[:22,:],axis=0) + 0.05*np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    for n in range(nTVB-nBands*nCells,nTVB):
        ax.plot(kList[:,0],e_[:,n],color='r',lw=0.5)
        ax.scatter(kList[:,0],e_[:,n],s=weights[:,n]*50,color='b')
    plt.show()

""" Moiré functions """
def diagonalize_matrix(*args,machine='loc'):
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
def import_monolayer_parameters(monolayer_type,machine):
    """Import monolayer parameters, either DFT or fit ones."""
    hopping = {}
    epsilon = {}
    HSO = {}
    offset = {}
    for TMD in cfs.TMDs:
        temp = np.load(get_home_dn(machine)+'Inputs/tb_'+TMD+'.npy') if monolayer_type=='fit' else np.array(cfs.initial_pt[TMD])
        hopping[TMD] = cfs.find_t(temp)
        epsilon[TMD] = cfs.find_e(temp)
        HSO[TMD] = cfs.find_HSO(temp[-2:])
        offset[TMD] = temp[-3]
    return (hopping,epsilon,HSO,offset)
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

""" Spreading """
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

""" Home directory name """
def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/Code/bilayer_v2.0/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/bilayer_v2.0/'
    elif machine == 'maf':
        return '/users/rossid/bilayer_v2.0/'

""" LDOS FUNCTIONS """
def LDOS_kList(kPts,theta,center='G'):
    """ Here we have to define the grid of momentum points to sum over.
    Here now is a grid of the mini-BZ.
    """
    G_M = cfs.get_reciprocal_moire(theta/180*np.pi)     #7 reciprocal moire lattice vectors
    G1,G2 = G_M[1:3]
    kList = np.zeros((kPts,kPts,2))
    if center=='K':
        b2 = 4*np.pi/np.sqrt(3)/cfs.dic_params_a_mono['WSe2'] * np.array([0,1])
        b1 = cfs.R_z(-np.pi/3) @ b2
        b6 = cfs.R_z(-2*np.pi/3) @ b2
        centerPoint = (b1+b6)/3
    elif center=='G':
        centerPoint = np.zeros(2)
    for ix in range(kPts):
        for iy in range(kPts):
            kList[ix,iy] = centerPoint + G1*ix/kPts + G2*iy/kPts
    kFlat = kList.reshape(-1,2)
    return kFlat

def LDOS_rList(rPts,theta):
    """ Here we have to define the real space points to compute.
    We take a cut: W/W -> Se/W -> W/S -> W/W
    """
    a1,a2 = get_rs_moire(theta)        #real space moirè vectors
    rList = np.zeros((rPts,2))
    for i in range(rPts):
        rList[i] = (a1+a2)*i/rPts
    return rList

def get_rs_moire(theta):
    """
    Here we compute the moirè real-space vectors from the reciprocal ones.
    """
    G_M = cfs.get_reciprocal_moire(theta/180*np.pi)     #7 reciprocal moire lattice vectors
    a_M = 2*np.pi*np.linalg.inv(G_M[1:3])
    a1 = a_M[:,0]
    a2 = cfs.R_z(np.pi/3) @ a1
    return [a1,a2]

def lorentzian(E, E0, eta):
    return eta / (np.pi * ((E - E0)**2 + eta**2))

def compute_LDOS(evals,evecs,*args):
    """
    Computation of LDOS over real space.
    """
    rList, eList, nShells, nCells, theta, kFlat, spreadE = args
    kPts = kFlat.shape[0]
    rPts = rList.shape[0]
    ePts = len(eList)

    LDOS = np.zeros((rPts,ePts))
    lu = lu_table(nShells)
    Kbs = np.zeros((nCells,2))
    G_M = cfs.get_reciprocal_moire(theta/180*np.pi)     #7 reciprocal moire lattice vectors
    for i in range(nCells):
        Kbs[i] = G_M[1]*lu[i][0] + G_M[2]*lu[i][1]

    ig = np.arange(nCells)[np.newaxis, :]  # (1, nCells)
    alpha = np.arange(44)[:, np.newaxis]  # (44, 1)       #index over alphas -> orbitals
    ind = (alpha % 22) + ig * 22 + nCells * 22 * (alpha // 22)  #44,nCells
    for ik in tqdm(range(kPts), desc="Momentum iteration"):
        evals_k = evals[ik]
        evecs_k = evecs[ik]
        kGs = Kbs + kFlat[ik]        #nCells,2
        phases = np.exp(1j * rList @ kGs.T)[np.newaxis,:,:]     #1,nR,nCells
        for n,En in enumerate(evals_k):
            #Vectorized calculation
            coeffs = evecs_k[ind,n]
            coeffs_all = coeffs[:,np.newaxis,:]  # (44, 1, nCells)
            #
            psi_alpha = np.sum(phases * coeffs_all, axis=-1)  # (44, nR)
            psi_r_all = np.sum(np.abs(psi_alpha)**2, axis=0)  # nR
            lorentz_matrix = lorentzian(eList, En, spreadE)  # nE
            LDOS += psi_r_all[:,None] * lorentz_matrix[None,:] / kPts # (nR, nE)
    return LDOS







