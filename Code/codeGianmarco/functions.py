import numpy as np
import scipy
import CORE_functions as cfs
import itertools
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.special import wofz      #for fitting weights in EDC
#import lmfit      #for fitting weights in EDC

machine = cfs.get_machine(os.getcwd())

def import_monolayer_parameters(monolayer_type,machine):
    """Import monolayer parameters, either DFT or fit ones."""
    hopping = {}
    epsilon = {}
    HSO = {}
    offset = {}
    for TMD in cfs.TMDs:
        temp = np.load('Inputs/tb_'+TMD+'.npy')
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
        return weight/(k_grid**2+K2)/((e_grid-E_temp)**2+E2)
    elif type_of_spread == 'Gauss':
        return weight*np.exp(-(k_grid/spread_k)**2)*np.exp(-((e_grid-E_temp)/spread_E)**2)


