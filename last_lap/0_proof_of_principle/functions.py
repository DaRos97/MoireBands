import numpy as np
from PIL import Image

materials = ['WSe2','WS2']

#Monolayer lattice lengths, in Angstrom
dic_params_a_mono = {
        'WS2': 3.18,
        'WSe2': 3.32,
        }

a_1 = np.array([1,0])
a_2 = np.array([-1/2,np.sqrt(3)/2])
list_ind = {'P': [0,1,2,3,4,5], 'AP':[0,2,4]}

list_f = {  'P':    np.linspace(0,0.1,20),
            'AP':   np.linspace(0,0.1,10)
            }

def big_H(K_,lu,all_pars,G_M):
    """Computes the large Hamiltonian containing all the moire replicas.

    """
    type_of_stacking,m1,m2,mu,a,b,c,f1,f2,N,V,phi = all_pars
    n_cells = int(1+3*N*(N+1))          #Number of mBZ copies
    H_up = np.zeros((n_cells,n_cells),dtype=complex)
    H_down = np.zeros((n_cells,n_cells),dtype=complex)
    H_int = np.zeros((n_cells,n_cells),dtype=complex)
    #Diagonal parts
    for n in range(n_cells):
        Kn = K_ + G_M[0]*lu[n][0] + G_M[1]*lu[n][1]
        H_up[n:(n+1),n:(n+1)] = -np.linalg.norm(Kn)**2/2/m1
        H_down[n:(n+1),n:(n+1)] = -np.linalg.norm(Kn)**2/2/m2+c
        H_int[n:(n+1),n:(n+1)] = get_t0(Kn,a,b,type_of_stacking)
    #Moirè part
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
                H_up[s,nn] = get_V(V,phi,g)
                H_down[s,nn] = get_V(V,phi,g)
                H_int[s,nn] = get_t1(f1,f2,g)
    #All together
    final_H = np.zeros((2*n_cells,2*n_cells),dtype=complex)
    final_H[:n_cells,:n_cells] = H_up
    final_H[n_cells:,n_cells:] = H_down
    final_H[n_cells:,:n_cells] = H_int
    final_H[:n_cells,n_cells:] = np.conjugate(H_int.T)
    #Chemical potential
    final_H += np.identity(2*n_cells)*mu
    return final_H

def get_t0(Kn,a,b,type_of_stacking):
    res = a
    for i in list_ind[type_of_stacking]:
        res += b*np.exp(1j*np.dot(Kn,np.dot(R_z(np.pi/3*i),a_1))*dic_params_a_mono['WSe2'])
    return res

def get_t1(f1,f2,ind):
    return f1 if ind%2 == 0 else f2

def get_V(V,phi,ind):
    return V*np.exp(1j*(-1)**(ind%2)*phi)

def get_K(cut,n_pts):
    res = np.zeros((n_pts,2))
    a_mono = dic_params_a_mono['WSe2']
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

def extract_png(fig_fn,cut_bounds):
    pic_0 = np.array(np.asarray(Image.open(fig_fn)))
    #We go from -1 to 1 in image K cause the picture is stupid
    Ki = -1.4
    Kf = 1.4
    Ei = 0
    Ef = -3.5
    #Empirically extracted for S11 from -1 to +1
    P_ki = 810
    P_kf = 2370
    p_len = int((P_kf-P_ki)/2*(Kf-Ki))   #number of pixels from Ki to Kf
    p_ki = int((P_ki+P_kf)//2 - p_len//2)
    p_kf = int((P_ki+P_kf)//2 + p_len//2)
    #
    p_ei = 85       #correct
    p_ef = 1908     #correct
    if len(cut_bounds) == 4:#Image cut
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

def get_Moire(a_M):     
    """Compute Moire reciprocal lattice vectors.

    """
    G_M = [0,4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
    G_M[0] = np.tensordot(R_z(-np.pi/3),G_M[1],1)
    return G_M

def R_z(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

def fn_list(list_):
    fn = ''
    for i in list_:
        if type(i)==str:
            fn += i
        elif type(i) in [int,np.int64]:
            fn += str(i)
        elif type(i) in [float,np.float64]:
            fn += "{:.4f}".format(i)
        if list_.index(i) != len(list_)-1:
            fn += '_'
    return fn

def get_energies_fn(all_pars,machine):
    return get_res_dn(machine) + 'raw/energies_'+ fn_list(all_pars) + '.npy'

def get_weights_fn(all_pars,machine):
    return get_res_dn(machine) + 'raw/weights_' + fn_list(all_pars) + '.npy'

def get_spread_fn(all_pars,pars_spread,machine):
    return get_res_dn(machine) + 'raw/spread_'+ fn_list(all_pars) + fn_list(pars_spread) + '.npy'

def get_fig_fn(all_pars,pars_spread,machine):
    return get_res_dn(machine) + 'figures/fig_'+ fn_list(all_pars) + fn_list(pars_spread) + '.png'

def get_S11_fn(machine):
    return get_home_dn(machine)+'inputs/S11_KGK_WSe2onWS2_v1.png'

def get_S3_fn(machine):
    return get_home_dn(machine)+'inputs/S3_KGK_WSe2onWS2_v1.png'

def get_res_dn(machine):
    return get_home_dn(machine)+'results/'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/last_lap/0_proof_of_principle/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/0_proof_of_principle/'
    elif machine == 'maf':
        pass

def get_machine(cwd):
    """Selects the machine the code is running on by looking at the working directory. Supports local, hpc (baobab or yggdrasil) and mafalda.

    Parameters
    ----------
    pwd : string
        Result of os.pwd(), the working directory.

    Returns
    -------
    string
        An acronim for the computing machine.
    """
    if cwd[6:11] == 'dario':
        return 'loc'
    elif cwd[:20] == '/home/users/r/rossid':
        return 'hpc'
    elif cwd[:13] == '/users/rossid':
        return 'maf'

def tqdm(x):
    return x

