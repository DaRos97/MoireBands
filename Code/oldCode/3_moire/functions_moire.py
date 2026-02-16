import numpy as np
import CORE_functions as cfs
from PIL import Image
import itertools

def get_pars(ind):
    lMonolayer_type = ['fit',]
    lInterlayer_symm = ['C6',]
    pars_Vgs = [0.007,]        #Moire potential at Gamma
    pars_Vks = [0.0077,]             #Moire potential at K
    phi_G = [np.pi,]                #Phase at Gamma
    phi_K = [-106*np.pi/180,]       #Phase at K
    lSample = ['S3',]  #this and theta are related! Sample needed also for interlayer parameters' choice
    lTheta = [2.8,] if lSample[0]=='S11' else [1.8,]    #twist angle
    lN = [1,]                       #number of BZ circles
    lCuts = ['K-G-Kp',]          #'Kp-G-K-Kp',]
    lKpts = [1002,]
    lWeights = [1,]              #Weights of intensities -> 1 is correct, 0.5 has more comparison with exp
    #
    ll = [lMonolayer_type,lInterlayer_symm,pars_Vgs,pars_Vks,phi_G,phi_K,lTheta,lSample,lN,lCuts,lKpts,lWeights]
    return list(itertools.product(*ll))[ind]

def big_H(K_,lu,pars_monolayer,pars_interlayer,pars_moire):
    """Computes the large Hamiltonian containing all the moire replicas.
    Each k-point has 44*n_mBZ dimension, with n_mBZ the number of mini-BZ considered.
    The basis is: 0 to 22*n_mBZ-1 -> WSe2, 22*n_mBZ to 44*n_mBZ-1 -> WS2.
        0 to 21 has the basis of the monolayer for mini BZ #0, and so on for the other mBZ.

    """
    hopping,epsilon,HSO,offset = pars_monolayer
    N,n_cells,pars_V,G_M,H_moires = pars_moire
    #
    H_up = np.zeros((n_cells*22,n_cells*22),dtype=complex)
    H_down = np.zeros((n_cells*22,n_cells*22),dtype=complex)
    H_int = np.zeros((n_cells*22,n_cells*22),dtype=complex)
    #
    args_WSe2 = (hopping['WSe2'],epsilon['WSe2'],HSO['WSe2'],cfs.dic_params_a_mono['WSe2'],offset['WSe2'])
    args_WS2 = (hopping['WS2'],epsilon['WS2'],HSO['WS2'],cfs.dic_params_a_mono['WS2'],offset['WS2'])
    for n in range(n_cells):
        Kn = K_ + G_M[0]*lu[n][0] + G_M[1]*lu[n][1]
        H_up[n*22:(n+1)*22,n*22:(n+1)*22] = cfs.H_monolayer(Kn,*args_WSe2)
        H_down[n*22:(n+1)*22,n*22:(n+1)*22] = cfs.H_monolayer(Kn,*args_WS2)+ H_interlayer_c(pars_interlayer) #interlayer c just on WS2
        H_int[n*22:(n+1)*22,n*22:(n+1)*22] = H_interlayer(Kn,pars_interlayer)   #interlayer -> a and b
    #Moirè replicas
    m = [[-1,1],[-1,0],[0,-1],[1,-1],[1,0],[0,1]]
    for n in range(0,N+1):      #Circles
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       #Indices inside the circle
            ind_s = lu[s]
            for i in m:
                ind_nn = (ind_s[0]+i[0],ind_s[1]+i[1])  #nn-> nearest neighbour
                try:
                    nn = lu.index(ind_nn)
                except:
                    continue
                g = m.index(i)
                H_up[s*22:(s+1)*22,nn*22:(nn+1)*22] = H_moires[g%2]    #H_moire(g,pars_moire[1])
                H_down[s*22:(s+1)*22,nn*22:(nn+1)*22] = H_moires[g%2]  #H_moire(g,pars_moire[1])
    #All together
    final_H = np.zeros((2*n_cells*22,2*n_cells*22),dtype=complex)
    final_H[:n_cells*22,:n_cells*22] = H_up
    final_H[n_cells*22:,n_cells*22:] = H_down
    final_H[n_cells*22:,:n_cells*22] = H_int
    final_H[:n_cells*22,n_cells*22:] = H_int.T.conj()
    #Global offset due to interlayer
    final_H += np.identity(2*n_cells*22)*pars_interlayer[1][-1]
    return final_H

def H_interlayer_c(pars_interlayer):
    H_int_c = np.zeros((22,22))
    H_int_c[8,8] = pars_interlayer[1][2]
    H_int_c[8+11,8+11] = pars_interlayer[1][2]
    return H_int_c

def H_interlayer(k_,pars_interlayer):
    H_int_res = np.zeros((22,22),dtype=complex)
    if pars_interlayer[0]=='U1':
        t_k = -pars_interlayer[1][0] + pars_interlayer[1][1]*np.linalg.norm(k_)**2
    elif pars_interlayer[0]=='C6':
        aa = cfs.dic_params_a_mono['WSe2']
        arr0 = aa*np.array([1,0])
        t_k = -pars_interlayer[1][0]
        for i in range(6):
            t_k += pars_interlayer[1][1]*np.exp(1j*k_@cfs.R_z(np.pi/3*i)@arr0)
    elif pars_interlayer[0]=='C3':
        aa = cfs.dic_params_a_mono['WSe2']
        arr0 = aa*np.array([1,0])/np.sqrt(3)
        t_k = 0
        for i in range(3):
            t_k += pars_interlayer[1][1]*np.exp(1j*k_@cfs.R_z(2*np.pi/3*i)@arr0)
    elif pars_interlayer[0]=='no':
        t_k = 0
    ind_pze = 8     #index of p_z(even) orbital
    for i in range(2):
        H_int_res[ind_pze+11*i,ind_pze+11*i] = t_k
    return H_int_res

def H_moire(g,pars_V):          #g is a integer from 0 to 5
    """Compute moire interlayer potential. 
    Distinguis in- and out-of- plane orbitals.
    """
    V_G,V_K,psi_G,psi_K = pars_V
    Id = np.zeros((22,22),dtype = complex)
    out_of_plane = V_G*np.exp(1j*(-1)**(g%2)*psi_G)
    in_plane = V_K*np.exp(1j*(-1)**(g%2)*psi_K)
    list_out = (0,1,2,5,8)      #out-of-plane orbitals (all ones containing a z)
    list_in = (3,4,6,7,9,10)    #in-plane orbitals 
    for i in list_out:
        Id[i,i] = out_of_plane
        Id[i+11,i+11] = out_of_plane
    for i in list_in:
        Id[i,i] = in_plane
        Id[i+11,i+11] = in_plane
    return Id

def extract_png(fig_fn,cut_bounds,sample):
    pic_0 = np.array(np.asarray(Image.open(fig_fn)))
    #We go from -1 to 1 in image K cause the picture is stupid
    Ki, Kf, Ei, Ef, P_ki, P_kf, p_ei, p_ef = cfs.dic_pars_samples[sample]
    #Empirically extracted for sample from -1 to +1
    p_len = int((P_kf-P_ki)/2*(Kf-Ki))   #number of pixels from Ki to Kf
    p_ki = int((P_ki+P_kf)//2 - p_len//2)
    p_kf = int((P_ki+P_kf)//2 + p_len//2)
    #
    if len(cut_bounds) == 4:#cut image
        ki,kf,ei,ef = cut_bounds
        pc_lenk = int(p_len/(Kf-Ki)*(kf-ki)) #number of pixels in cut image
        pc_ki = int((p_ki+p_kf)//2-pc_lenk//2)
        pc_kf = int((p_ki+p_kf)//2+pc_lenk//2)
        #
        pc_lene = int((p_ef-p_ei)/(Ei-Ef)*(ei-ef))
        pc_ei = p_ei + int((p_ef-p_ei)/(Ei-Ef)*(Ei-ei))
        pc_ef = p_ei + int((p_ef-p_ei)/(Ei-Ef)*(Ei-ef))
        return pic_0[pc_ei:pc_ef,pc_ki:pc_kf]
    else:
        return pic_0[p_ei:p_ef,p_ki:p_kf]

def lu_table(N):
    """Computes the look-up table for the index of the mini-BZ in terms of the 
    reciprocal lattice vector indexes

    """
    n_cells = int(1+3*N*(N+1))
    lu = []
    m = [[-1,1],[-1,0],[0,-1],[1,-1],[1,0],[0,1]]
    for n in range(0,N+1):      #circles go from 0 (central BZ) to N included
        i = 0
        j = 0
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       #mini-BZ index
            if s == np.sign(n)*(1+(n-1)*n*3):
                lu.append((n,0))
            else:
                lu.append((lu[-1][0]+m[i][0],lu[-1][1]+m[i][1]))
                if j == n-1:
                    i += 1
                    j = 0
                else:
                    j += 1
    return lu

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

def get_reciprocal_moire(theta):
    """Compute moire reciprocal lattice vectors.
    They depend on the moire length for the size and on the orientation of the mini-BZ for the direction.
    """
    G_M = [0,np.matmul(cfs.R_z(cfs.miniBZ_rotation(theta)),4*np.pi/np.sqrt(3)/cfs.moire_length(theta)*np.array([0,1]))]
    G_M[0] = np.matmul(cfs.R_z(-np.pi/3),G_M[1])
    return G_M

def get_list_fn(l):
    fn = ''
    for i in range(len(l)):
        fn += "{:.4f}".format(l[i])
        if i != len(l)-1:
            fn += '_'
    return fn

def get_spread_fn(DFT,N,pars_V,pixel_factor,a_M,interlayer_type,pars_spread,weight_exponent,machine):
    name_v = get_list_fn(pars_V)
    name_sp = get_list_fn(pars_spread[:2])
    txt_dft = 'DFT' if DFT else 'fit'
    return get_home_dn(machine)+'results/E_data/spread_'+txt_dft+"{:.1f}".format(weight_exponent)+'_'+pars_spread[-1]+'_'+name_sp+'_'+str(N)+'_'+name_v+'_'+str(pixel_factor)+'_'+"{:.1f}".format(a_M)+'_'+interlayer_type+'.npy'

def get_fig1_fn(DFT,N,pars_V,p_f,a_M,interlayer_type,machine):
    name_v = get_list_fn(pars_V)
    txt_dft = 'DFT' if DFT else 'fit'
    return get_home_dn(machine)+'results/figures/bands/bands_'+txt_dft+'_'+str(N)+'_'+name_v+'_'+str(p_f)+'_'+"{:.1f}".format(a_M)+'_'+interlayer_type+'.png'

def get_fig_fn(DFT,N,pars_V,p_f,a_M,interlayer_type,pars_spread,machine):
    name_v = get_list_fn(pars_V)
    name_sp = get_list_fn(pars_spread[:2])
    txt_dft = 'DFT' if DFT else 'fit'
    return get_home_dn(machine)+'results/figures/spread/'+txt_dft+'_'+pars_spread[-1]+'_'+name_sp+'_'+str(N)+'_'+name_v+'_'+str(p_f)+'_'+"{:.1f}".format(a_M)+'_'+interlayer_type+'.png'

def get_data_fns(pars_data,pars_spread,machine):
    """
    Filename of data: energy-weights and spreading.
    Spread needs additional parameters.
    """
    monolayer_type, interlayer_symmetry, Vg, Vk, phiG, phiK, theta, sample, N, cut, k_pts, weight_exponent = pars_data
    spread_k,spread_E,type_spread,deltaE,E_min,E_max = pars_spread
    common_name = '_'+monolayer_type+'_'+interlayer_symmetry+'_'+"{:.5f}".format(Vg)+'-'+"{:.5f}".format(Vk)+'-'+"{:.5f}".format(phiG)+'-'+"{:.5f}".format(phiK)+'_'+"{:.2f}".format(theta)+'_'+str(N)+'_'+cut+'_'+str(k_pts)
    spread_name = '_'+"{:.5f}".format(spread_k)+'_'+"{:.5f}".format(spread_E)+'_'+type_spread+'_'+"{:3f}".format(deltaE)+'_'+"{:.2f}".format(E_min)+'_'+"{:.2f}".format(E_max)
    data_dn = get_results_dn(machine)
    result = []
    for t in ['en_wh','spread']:
        result.append(data_dn+t+common_name)
        if t=='spread':
            result[-1] += spread_name
        result[-1] += '.npy'
    result.append(get_home_dn(machine)+'Figures/spread'+common_name+spread_name+'_'+"{:.3f}".format(weight_exponent)+'.png')
    return result

def get_sample_fn(sample,machine,zoom=False):
    v = 'v2' if sample == 'S3' else 'v1'
    v = 'zoom1' if zoom else v
    return get_inputs_dn(machine)+sample+'_KGK_WSe2onWS2_'+v+'.png'

def get_pars_mono_fn(TMD,machine,monolayer_type='DFT'):
    return get_inputs_dn(machine)+'pars_'+TMD+'.npy'

def get_SOC_fn(TMD,machine):
    return get_inputs_dn(machine)+TMD+'_SOC.npy'

def get_pars_interlayer_fn(sample,interlayer_type,monolayer_type,machine):
    int_fn = sample+'_'+monolayer_type+'_'+interlayer_type+'_pars_interlayer.npy'
    return get_inputs_dn(machine)+int_fn

def get_inputs_dn(machine):
    return get_home_dn(machine)+'inputs/'

def get_results_dn(machine):
    return get_home_dn(machine)+'Data/'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/Code/3_moire/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/3_moire/'
    elif machine == 'maf':
        return '/users/rossid/3_moire/'

def import_monolayer_parameters(monolayer_type,machine):
    """Import monolayer parameters, either DFT or fit ones."""
    hopping = {}
    epsilon = {}
    HSO = {}
    offset = {}
    for TMD in cfs.TMDs:
        temp = np.load(get_pars_mono_fn(TMD,machine)) if monolayer_type=='fit' else np.array(cfs.initial_pt[TMD])
        hopping[TMD] = cfs.find_t(temp)
        epsilon[TMD] = cfs.find_e(temp)
        HSO[TMD] = cfs.find_HSO(temp[-2:])
        offset[TMD] = temp[-3]
    return (hopping,epsilon,HSO,offset)

def import_moire_parameters(N,pars_V,theta):
    """ Import moire parameters.
    Moirè potential of bilayer is different at Gamma (d_z^2 orbital) and K (d_xy orbitals).
    Gamma point does not have a DFT estimate. K point value from Louk's paper: L.Rademaker, Phys. Rev. B 105, 195428 (2022)
    Hamiltonian of Moire interlayer (diagonal with correct signs of phase). Compute it here because is k-independent.
    """
    n_cells = int(1+3*N*(N+1))
    G_M = get_reciprocal_moire(theta/180*np.pi)
    H_moires = [H_moire(0,pars_V),H_moire(1,pars_V)]
    return (N,n_cells,pars_V,G_M,H_moires)
















