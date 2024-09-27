import numpy as np
import CORE_functions as cfs
from PIL import Image
import itertools


def get_pars(ind):
    DFT = [True,False]
    samples = ['S11','S3']
    int_types = ['U1','C6','C3',] #1
    pars_Vgs = [0.02,0.03,0.04,0.05]   #5
    pars_Vks = [0.01,0.02]  #5
    phi_G = [np.pi,]
    phi_K = [-106*2*np.pi/360,]
    #
    ll = [samples,int_types,pars_Vgs,pars_Vks,phi_G,phi_K]
    combs = list(itertools.product(*ll))
    #
    return combs[ind] 

def big_H(K_,lu,pars_monolayer,pars_interlayer,pars_moire):
    """Computes the large Hamiltonian containing all the moire replicas.

    """
    hopping,epsilon,HSO,offset = pars_monolayer
    N,pars_V,G_M,Ham_moire = pars_moire
    #
    n_cells = int(1+3*N*(N+1))          #Number of mBZ copies
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
    #Moirè
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
                H_up[s*22:(s+1)*22,nn*22:(nn+1)*22] = Ham_moire[g%2]    #H_moire(g,pars_moire[1])
                H_down[s*22:(s+1)*22,nn*22:(nn+1)*22] = Ham_moire[g%2]  #H_moire(g,pars_moire[1])
    #All together
    final_H = np.zeros((2*n_cells*22,2*n_cells*22),dtype=complex)
    final_H[:n_cells*22,:n_cells*22] = H_up
    final_H[n_cells*22:,n_cells*22:] = H_down
    final_H[n_cells*22:,:n_cells*22] = H_int
    final_H[:n_cells*22,n_cells*22:] = np.conjugate(H_int.T)
    #Global offset due to interlayer
    final_H += np.identity(2*n_cells*22)*pars_interlayer[1][-1]
    return final_H

def H_interlayer_c(pars_interlayer):
    H = np.zeros((22,22))    
    H[8,8] = H[8+11,8+11] = pars_interlayer[1][2]
    return H

def H_interlayer(k_,pars_interlayer):
    H = np.zeros((22,22),dtype=complex)
    if pars_interlayer[0]=='U1':
        t_k = -pars_interlayer[1][0] + pars_interlayer[1][1]*np.linalg.norm(k_)**2
    elif pars_interlayer[0]=='C6':
        aa = cfs.dic_params_a_mono['WSe2']
        t_k = -pars_interlayer[1][0] + pars_interlayer[1][1]*2*(np.cos(k_[0]*aa)+np.cos(k_[0]/2*aa)*np.cos(np.sqrt(3)/2*k_[1]*aa))
    elif pars_interlayer[0]=='C3':
        aa = cfs.dic_params_a_mono['WSe2']
        delta = aa*np.array([np.array([1,0]),np.array([1/2,np.sqrt(3)/2]),np.array([-1/2,np.sqrt(3)/2])])
        t_k = 0
        for i in range(3):
            t_k += pars_interlayer[1][1]*np.exp(1j*np.dot(k_,delta[i]))
    elif pars_interlayer[0]=='no':
        t_k = 0
    ind_pze = 8
    for i in range(2):
        H[ind_pze+11*i,ind_pze+11*i] = t_k 
    return H

def H_moire(g,pars_V):          #g is a integer from 0 to 5
    """Compute moire interlayer potential. 
    Distinguis in- and out-of- plane orbitals.
    """
    V_G,V_K,psi_G,psi_K = pars_V
    Id = np.zeros((22,22),dtype = complex)
    out_of_plane = V_G*np.exp(1j*(-1)**(g%2)*psi_G)
    in_plane = V_K*np.exp(1j*(-1)**(g%2)*psi_K)
    list_out = (0,1,2,5,8)
    list_in = (3,4,6,7,9,10)
    for i in list_out:
        Id[i,i] = out_of_plane
        Id[i+11,i+11] = out_of_plane
    for i in list_in:
        Id[i,i] = in_plane
        Id[i+11,i+11] = in_plane
    return Id

def get_K(cut,n_pts):
    res = np.zeros((n_pts,2))
    a_mono = cfs.dic_params_a_mono['WSe2']
    if cut == 'KGK':
        K = np.array([4*np.pi/3,0])/a_mono
        for i in range(n_pts):
            res[i,0] = K[0]/(n_pts//2)*(i-n_pts//2)
    if cut == 'KMKp':
        M = np.array([np.pi,np.pi/np.sqrt(3)])/a_mono
        K = np.array([4*np.pi/3,0])/a_mono
        Kp = np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])/a_mono
        for i in range(n_pts//2):
            res[i] = K + (M-K)*i/(n_pts//2)
        for i in range(n_pts//2,n_pts):
            res[i] = M + (Kp-M)*i/(n_pts//2)
    return res

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

def weight_spreading(weight,K,E,k_grid,e_grid,pars_spread):
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
    spread_K,spread_E,type_of_spread = pars_spread
    if type_of_spread == 'Lorentz':
        E2 = spread_E**2
        K2 = spread_K**2
        return weight/((k_grid-K)**2+K2)/((e_grid-E)**2+E2)
    elif type_of_spread == 'Gauss':
        return weight*np.exp(-((k_grid-K)/spread_K)**2)*np.exp(-((e_grid-E)/spread_E)**2)

def normalize_spread(spread,k_pts,e_pts):
    #Transform lor to a png formati. in the range of white/black of the original picture
    max_lor = np.max(np.ravel(spread))
    min_lor = np.min(np.ravel(np.nonzero(spread)))
    whitest = 255
    blackest = 0     
    normalized_lor = np.zeros((k_pts,e_pts))
    for i in range(k_pts):
        for j in range(e_pts):
            normalized_lor[i,j] = int((whitest-blackest)*(1-spread[i,j]/(max_lor-min_lor))+blackest)
    picture = np.flip(normalized_lor.T,axis=0)   #invert e-axis to have the same structure
    return picture

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

def get_spread_fn(DFT,N,pars_V,p_f,a_M,interlayer_type,pars_spread,machine):
    name_v = get_list_fn(pars_V)
    name_sp = get_list_fn(pars_spread[:2])
    txt_dft = 'DFT' if DFT else 'fit'
    return get_home_dn(machine)+'results/data/spread_'+txt_dft+'_'+pars_spread[-1]+'_'+name_sp+'_'+str(N)+'_'+name_v+'_'+str(p_f)+'_'+"{:.1f}".format(a_M)+'_'+interlayer_type+'.npy'

def get_energies_fn(DFT,N,pars_V,p_f,a_M,interlayer_type,machine):
    name_v = get_list_fn(pars_V)
    txt_dft = 'DFT' if DFT else 'fit'
    return get_home_dn(machine)+'results/data/energies_'+txt_dft+'_'+str(N)+'_'+name_v+'_'+str(p_f)+'_'+"{:.1f}".format(a_M)+'_'+interlayer_type+'.npy'

def get_weights_fn(DFT,N,pars_V,p_f,a_M,interlayer_type,machine):
    name_v = get_list_fn(pars_V)
    txt_dft = 'DFT' if DFT else 'fit'
    return get_home_dn(machine)+'results/data/weights_'+txt_dft+'_'+str(N)+'_'+name_v+'_'+str(p_f)+'_'+"{:.1f}".format(a_M)+'_'+interlayer_type+'.npy'

def get_fig1_fn(DFT,N,pars_V,p_f,a_M,interlayer_type,machine):
    name_v = get_list_fn(pars_V)
    txt_dft = 'DFT' if DFT else 'fit'
    return get_home_dn(machine)+'results/figures/bands/bands_'+txt_dft+'_'+str(N)+'_'+name_v+'_'+str(p_f)+'_'+"{:.1f}".format(a_M)+'_'+interlayer_type+'.png'

def get_fig_fn(DFT,N,pars_V,p_f,a_M,interlayer_type,pars_spread,machine):
    name_v = get_list_fn(pars_V)
    name_sp = get_list_fn(pars_spread[:2])
    txt_dft = 'DFT' if DFT else 'fit'
    return get_home_dn(machine)+'results/figures/spread/'+txt_dft+'_'+pars_spread[-1]+'_'+name_sp+'_'+str(N)+'_'+name_v+'_'+str(p_f)+'_'+"{:.1f}".format(a_M)+'_'+interlayer_type+'.png'

def get_sample_fn(sample,machine,zoom=False):
    v = 'v2' if sample == 'S3' else 'v1'
    v = 'zoom1' if zoom else v
    return get_home_dn(machine)+'inputs/'+sample+'_KGK_WSe2onWS2_'+v+'.png'

def get_pars_mono_fn(TMD,machine,dft=False):
    get_dft = '_DFT' if dft else '_fit'
    return get_home_dn(machine)+'inputs/pars_'+TMD+get_dft+'.npy'

def get_pars_interlayer_fn(sample,interlayer_type,DFT,machine):
    txt = 'DFT' if DFT else 'fit'
    int_fn = sample+'_'+txt+'_'+interlayer_type+'_pars_interlayer.npy'
    return get_home_dn(machine)+'inputs/'+int_fn

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/last_lap/3_moire/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/3_moire/'
    elif machine == 'maf':
        return '/users/rossid/3_moire/'

