import numpy as np

materials = ['WSe2','WS2']

#Monolayer lattice lengths, in Angstrom
dic_params_a_mono = {
        'WS2': 3.18,
        'WSe2': 3.32,
        }

a_1 = np.array([1,0])
a_2 = np.array([-1/2,np.sqrt(3)/2])
J_plus = ((3,5), (6,8), (9,11))
J_minus = ((1,2), (3,4), (4,5), (6,7), (7,8), (9,10), (10,11))
J_MX_plus = ((3,1), (5,1), (4,2), (10,6), (9,7), (11,7), (10,8))
J_MX_minus = ((4,1), (3,2), (5,2), (9,6), (11,6), (10,7), (9,8), (11,8))

def big_H(K_,lu,pars_monolayer,pars_interlayer,pars_moire):
    """Computes the large Hamiltonian containing all the moire replicas.

    """
    N,pars_V,G_M = pars_moire
    n_cells = int(1+3*N*(N+1))
    H_up = np.zeros((n_cells*22,n_cells*22),dtype=complex)
    H_down = np.zeros((n_cells*22,n_cells*22),dtype=complex)
    H_int = np.zeros((n_cells*22,n_cells*22),dtype=complex)
    #
    for n in range(n_cells):      #circles go from 0 (central BZ) to N included
        Kn = K_ + G_M[0]*lu[n][0] + G_M[1]*lu[n][1]
        H_up[n*22:(n+1)*22,n*22:(n+1)*22] = H_monolayer(Kn,pars_monolayer,'WSe2',pars_interlayer)   #interlayer -> d
        H_down[n*22:(n+1)*22,n*22:(n+1)*22] = H_monolayer(Kn,pars_monolayer,'WS2',pars_interlayer)  #interlayer -> c
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
                H_up[s*22:(s+1)*22,nn*22:(nn+1)*22] = H_moire(g,pars_moire[1])
                H_down[s*22:(s+1)*22,nn*22:(nn+1)*22] = H_moire(g,pars_moire[1])
    #All together
    final_H = np.zeros((2*n_cells*22,2*n_cells*22),dtype=complex)
    final_H[:n_cells*22,:n_cells*22] = H_up
    final_H[n_cells*22:,n_cells*22:] = H_down
    final_H[n_cells*22:,:n_cells*22] = H_int
    final_H[:n_cells*22,n_cells*22:] = np.conjugate(H_int.T)
    #Global offset due to interlayer
    final_H += np.identity(2*n_cells*22)*pars_interlayer[-1]
    return final_H

def H_monolayer(K_p,pars_H,TMD,pars_interlayer):
    """Monolayer Hamiltonian.
    TO CHECK

    """
    a_mono = dic_params_a_mono[TMD]
    t = pars_H[0][TMD]      #hopping
    epsilon = pars_H[1][TMD]
    HSO = pars_H[2][TMD]
    offset = pars_H[3][TMD]
    k_x,k_y = K_p       #momentum
    delta = a_mono*np.array([a_1, a_1+a_2, a_2, -(2*a_1+a_2)/3, (a_1+2*a_2)/3, (a_1-a_2)/3, -2*(a_1+2*a_2)/3, 2*(2*a_1+a_2)/3, 2*(a_2-a_1)/3])
    H_0 = np.zeros((11,11),dtype=complex)       #fist part without SO
    #Diagonal
    for i in range(11):
        H_0[i,i] += (epsilon[i] + 2*t[0][i,i]*np.cos(np.dot(K_p,delta[0])) 
                             + 2*t[1][i,i]*(np.cos(np.dot(K_p,delta[1])) + np.cos(np.dot(K_p,delta[2])))
                 )
    #Off diagonal symmetry +
    for ind in J_plus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (2*t[0][i,j]*np.cos(np.dot(K_p,delta[0])) 
                + t[1][i,j]*(np.exp(-1j*np.dot(K_p,delta[1])) + np.exp(-1j*np.dot(K_p,delta[2])))
                + t[2][i,j]*(np.exp(1j*np.dot(K_p,delta[1])) + np.exp(1j*np.dot(K_p,delta[2])))
                )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #Off diagonal symmetry -
    for ind in J_minus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (-2*1j*t[0][i,j]*np.sin(np.dot(K_p,delta[0])) 
                + t[1][i,j]*(np.exp(-1j*np.dot(K_p,delta[1])) - np.exp(-1j*np.dot(K_p,delta[2])))
                + t[2][i,j]*(-np.exp(1j*np.dot(K_p,delta[1])) + np.exp(1j*np.dot(K_p,delta[2])))
                )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #M-X coupling +
    for ind in J_MX_plus:
        i = ind[0]-1
        j = ind[1]-1
        temp = t[3][i,j] * (np.exp(1j*np.dot(K_p,delta[3])) - np.exp(1j*np.dot(K_p,delta[5])))
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #M-X coupling -
    for ind in J_MX_minus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (t[3][i,j] * (np.exp(1j*np.dot(K_p,delta[3])) + np.exp(1j*np.dot(K_p,delta[5])))
                   + t[4][i,j] * np.exp(1j*np.dot(K_p,delta[4]))
                   )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #Second nearest neighbor
    H_1 = np.zeros((11,11),dtype=complex)       #fist part without SO
    H_1[8,5] += t[5][8,5]*(np.exp(1j*np.dot(K_p,delta[6])) + np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,8] += np.conjugate(H_1[8,5])
    #
    H_1[10,5] += t[5][10,5]*(np.exp(1j*np.dot(K_p,delta[6])) - 1/2*np.exp(1j*np.dot(K_p,delta[7])) - 1/2*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,10] += np.conjugate(H_1[10,5])
    #
    H_1[9,5] += np.sqrt(3)/2*t[5][10,5]*(- np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,9] += np.conjugate(H_1[9,5])
    #
    H_1[8,7] += t[5][8,7]*(np.exp(1j*np.dot(K_p,delta[6])) - 1/2*np.exp(1j*np.dot(K_p,delta[7])) - 1/2*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[7,8] += np.conjugate(H_1[8,7])
    #
    H_1[8,6] += np.sqrt(3)/2*t[5][8,7]*(- np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,8] += np.conjugate(H_1[8,6])
    #
    H_1[9,6] += 3/4*t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,9] += np.conjugate(H_1[9,6])
    #
    H_1[10,6] += np.sqrt(3)/4*t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[7])) - np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,10] += np.conjugate(H_1[10,6])
    H_1[9,7] += H_1[10,6]
    H_1[7,9] += H_1[6,10]
    #
    H_1[10,7] += t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[6])) + 1/4*np.exp(1j*np.dot(K_p,delta[7])) + 1/4*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[7,10] += np.conjugate(H_1[10,7])
    #Combine the two terms
    H_TB = H_0 + H_1

    #### Spin orbit terms
    H = np.zeros((22,22),dtype = complex)
    H[:11,:11] = H_TB
    H[11:,11:] = H_TB
    H += HSO

    #Offset
    H += np.identity(22)*offset
    #Interlayer -> c
    if TMD == 'WS2':
        H[8,8] += pars_interlayer[2]
        H[8+11,8+11] += pars_interlayer[2]
    elif TMD == 'WSe2':     #-> d-factor on p_x(odd) orbitals
        H[3,3] += pars_interlayer[3]
        H[3+11,3+11] += pars_interlayer[3]
    return H

def H_interlayer(K_p,pars):
    H = np.zeros((22,22))
    H[8,8] = -pars[0] + pars[1]*np.linalg.norm(K_p)**2
    H[8+11,8+11] = -pars[0] + pars[1]*np.linalg.norm(K_p)**2
    return H

def H_moire(g,pars_V):          #g is a integer from 0 to 5
    """Compute moire interlayer potential. 

    """
    V_G,psi_G,V_K,psi_K = pars_V
    Id = np.zeros((22,22),dtype = complex)
    out_of_plane = V_G*np.exp(1j*(-1)**(g%2)*psi_G)
    in_plane = V_K*np.exp(1j*(-1)**(g%2)*psi_K)
    list_out = (0,1,2,5,8)
    list_in = (3,4,6,7,9,10)
    for i in list_out:
        Id[i,i] = Id[i+11,i+11] = out_of_plane
    for i in list_in:
        Id[i,i] = Id[i+11,i+11] = in_plane
    return Id

def find_t(dic_params_H):
    """Define hopping matrix elements from inputs and complete all symmetry related ones.

    """
    t = []
    t.append(np.zeros((11,11))) #t1
    t.append(np.zeros((11,11))) #t2
    t.append(np.zeros((11,11))) #t3
    t.append(np.zeros((11,11))) #t4
    t.append(np.zeros((11,11))) #t5
    t.append(np.zeros((11,11))) #t6
    #Independent parameters
    t[0][0,0] = dic_params_H[7]
    t[0][1,1] = dic_params_H[8]
    t[0][2,2] = dic_params_H[9]
    t[0][3,3] = dic_params_H[10]
    t[0][4,4] = dic_params_H[11]
    t[0][5,5] = dic_params_H[12]
    t[0][6,6] = dic_params_H[13]
    t[0][7,7] = dic_params_H[14]
    t[0][8,8] = dic_params_H[15]
    t[0][9,9] = dic_params_H[16]
    t[0][10,10] = dic_params_H[17]
    t[0][2,4] = dic_params_H[18]
    t[0][5,7] = dic_params_H[19]
    t[0][8,10] = dic_params_H[20]
    t[0][0,1] = dic_params_H[21]
    t[0][2,3] = dic_params_H[22]
    t[0][3,4] = dic_params_H[23]
    t[0][5,6] = dic_params_H[24]
    t[0][6,7] = dic_params_H[25]
    t[0][8,9] = dic_params_H[26]
    t[0][9,10] = dic_params_H[27]
    t[4][3,0] = dic_params_H[28]
    t[4][2,1] = dic_params_H[29]
    t[4][4,1] = dic_params_H[30]
    t[4][8,5] = dic_params_H[31]
    t[4][10,5] = dic_params_H[32]
    t[4][9,6] = dic_params_H[33]
    t[4][8,7] = dic_params_H[34]
    t[4][10,7] = dic_params_H[35]
    t[5][8,5] = dic_params_H[36]
    t[5][10,5] = dic_params_H[37]
    t[5][8,7] = dic_params_H[38]
    t[5][10,7] = dic_params_H[39]
    #Non-independent parameters
    list_1 = ((1,2,-1),(4,5,3),(7,8,6),(10,11,9))
    for inds in list_1:
        a,b,g = inds
        a -= 1
        b -= 1
        g -= 1
        t[1][a,a] = 1/4*t[0][a,a] + 3/4*t[0][b,b]
        t[1][b,b] = 3/4*t[0][a,a] + 1/4*t[0][b,b]
        t[1][a,b] = np.sqrt(3)/4*(t[0][a,a]-t[0][b,b]) - t[0][a,b]
        t[2][a,b] = -np.sqrt(3)/4*(t[0][a,a]-t[0][b,b]) - t[0][a,b]
        if g >= 0:
            t[1][g,g] = t[0][g,g]
            t[1][g,b] = np.sqrt(3)/2*t[0][g,a]-1/2*t[0][g,b]
            t[2][g,b] = -np.sqrt(3)/2*t[0][g,a]-1/2*t[0][g,b]
            t[1][g,a] = np.sqrt(3)/2*t[0][g,b]+1/2*t[0][g,a]
            t[2][g,a] = -np.sqrt(3)/2*t[0][g,b]+1/2*t[0][g,a]
    list_2 = ((1,2,4,5,3),(7,8,10,11,9))
    for inds in list_2:
        a,b,ap,bp,gp = inds
        a -= 1
        b -= 1
        ap -= 1
        bp -= 1
        gp -= 1
        t[3][ap,a] = 1/4*t[4][ap,a] + 3/4*t[4][bp,b]
        t[3][bp,b] = 3/4*t[4][ap,a] + 1/4*t[4][bp,b]
        t[3][bp,a] = t[3][ap,b] = -np.sqrt(3)/4*t[4][ap,a] + np.sqrt(3)/4*t[4][bp,b]
        t[3][gp,a] = -np.sqrt(3)/2*t[4][gp,b]
        t[3][gp,b] = -1/2*t[4][gp,b]
    t[3][8,5] = t[4][8,5]
    t[3][9,5] = -np.sqrt(3)/2*t[4][10,5]
    t[3][10,5] = -1/2*t[4][10,5]
    return t

def find_e(dic_params_H):
    """Define the array of on-site energies from inputs and symmetry related ones.

    """
    e = np.zeros(11)
    e[0] = dic_params_H[0]
    e[1] = e[0]
    e[2] = dic_params_H[1]
    e[3] = dic_params_H[2]
    e[4] = e[3]
    e[5] = dic_params_H[3]
    e[6] = dic_params_H[4]
    e[7] = e[6]
    e[8] = dic_params_H[5]
    e[9] = dic_params_H[6]
    e[10] = e[9]
    return e

def find_HSO(dic_params_H):
    """Compute the SO Hamiltonian. TO CHECK.

    """
    l_M = dic_params_H[40]
    l_X = dic_params_H[41]
    ####
    Mee_uu = np.zeros((6,6),dtype=complex)
    Mee_uu[1,2] = 1j*l_M
    Mee_uu[2,1] = -1j*l_M
    Mee_uu[4,5] = -1j*l_X/2
    Mee_uu[5,4] = 1j*l_X/2
    Mee_dd = -Mee_uu
    #
    Moo_uu = np.zeros((5,5),dtype=complex)
    Moo_uu[0,1] = -1j*l_M/2
    Moo_uu[1,0] = 1j*l_M/2
    Moo_uu[3,4] = -1j*l_X/2
    Moo_uu[4,3] = 1j*l_X/2
    Moo_dd = -Moo_uu
    #
    Moe_ud = np.zeros((5,6),dtype=complex)
    Moe_ud[0,0] = l_M*np.sqrt(3)/2
    Moe_ud[0,1] = 1j*l_M/2
    Moe_ud[0,2] = -l_M/2
    Moe_ud[1,0] = -1j*l_M*np.sqrt(3)/2
    Moe_ud[1,1] = -l_M/2
    Moe_ud[1,2] = -1j*l_M/2
    Moe_ud[2,4] = -l_X/2
    Moe_ud[2,5] = 1j*l_X/2
    Moe_ud[3,3] = l_X/2
    Moe_ud[4,3] = -1j*l_X/2
    Meo_du = np.conjugate(Moe_ud.T)
    #
    Meo_ud = np.zeros((6,5),dtype=complex)
    Meo_ud[0,0] = -l_M*np.sqrt(3)/2
    Meo_ud[0,1] = 1j*l_M*np.sqrt(3)/2
    Meo_ud[1,0] = -1j*l_M/2
    Meo_ud[1,1] = l_M/2
    Meo_ud[2,0] = l_M/2
    Meo_ud[2,1] = 1j*l_M/2
    Meo_ud[3,3] = -l_X/2
    Meo_ud[3,4] = 1j*l_X/2
    Meo_ud[4,2] = l_X/2
    Meo_ud[5,2] = -1j*l_X/2
    Moe_du = np.conjugate(Meo_ud.T)
    #
    Muu = np.zeros((11,11),dtype=complex)
    Muu[:5,:5] = Moo_uu
    Muu[5:,5:] = Mee_uu
    Mdd = np.zeros((11,11),dtype=complex)
    Mdd[:5,:5] = Moo_dd
    Mdd[5:,5:] = Mee_dd
    Mud = np.zeros((11,11),dtype=complex)
    Mud[:5,5:] = Moe_ud
    Mud[5:,:5] = Meo_ud
    Mdu = np.zeros((11,11),dtype=complex)
    Mdu[:5,5:] = Moe_du
    Mdu[5:,:5] = Meo_du
    #
    HSO = np.zeros((22,22),dtype=complex)
    HSO[:11,:11] = Muu
    HSO[11:,11:] = Mdd
    HSO[:11,11:] = Mud
    HSO[11:,:11] = Mdu
    ####
    return HSO

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
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       
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

def get_grid(pars_grid):
    """Compute the grid in momentum.

    """
    center, range_K, k_pts = pars_grid
    line = np.linspace(-range_K,range_K,k_pts)
    KX,KY = np.meshgrid(line,line)
    if center == 'K':
        KX += 4/3*np.pi/dic_params_a_mono['WSe2']
    return (KX,KY)

def spread_lor(K,sp_k,kx_list,ky_list):
    """Computes the Lorentzian spread.

    """
    return 1/((kx_list[:,None]-K[0])**2+(ky_list[None,:]-K[1])**2+sp_k**2)**(3/2)

def spread_gauss(K,sp_k,kx_list,ky_list):
    """Computes the Gaussian spread.

    """
    return np.exp(-((K[0]-kx_list[:,None])/sp_k)**2)*np.exp(-((K[1]-ky_list[None,:])/sp_k)**2)

spread_fun_dic = {'Lorentz':spread_lor, 'Gauss':spread_gauss}

def normalize_cut(en_cut,pars_grid):
    """Normalize the energy cut in grayscale and put it in imshow format.

    """
    center, range_K,k_pts = pars_grid
    max_lor = np.max(np.ravel(en_cut))
    min_lor = np.min(np.ravel(np.nonzero(en_cut)))
    whitest = 255
    blackest = 0     
    norm_lor = np.zeros(en_cut.shape)
    for i in range(k_pts):
        for j in range(k_pts):
            norm_lor[i,j] = int((whitest-blackest)*(1-en_cut[i,j]/(max_lor-min_lor))+blackest)
    return np.flip(norm_lor.T,axis=0)   #invert e-axis

def get_pars(ind):
    centers = ['G','K']
    DFTs = [True,False]
    lD = len(DFTs)
    pars_Vgs = [0.005,0.01,0.02,0.03]
    lVg = len(pars_Vgs)
    pars_Vks = [0.001,0.005,0.0077,0.01,0.015]
    lVk = len(pars_Vks)
    a_Moires = [79.8,70,60,50]
    laM = len(a_Moires)
    phi_g = np.pi
    phi_k = -106*2*np.pi/360
    #
    ind_c = ind//(lD*lVg*lVk*laM)
    ind_DFT = ind%(lD*lVg*lVk*laM) // (lVg*lVk*laM)
    ind_Vg = (ind%(lD*lVg*lVk*laM) % (lVg*lVk*laM)) // (lVk*laM)
    ind_Vk = ((ind%(lD*lVg*lVk*laM) % (lVg*lVk*laM)) % (lVk*laM)) // laM
    ind_aM = ((ind%(lD*lVg*lVk*laM) % (lVg*lVk*laM)) % (lVk*laM)) % laM
    return (centers[ind_c], DFTs[ind_DFT], [pars_Vgs[ind_Vg],phi_g,pars_Vks[ind_Vk],phi_k], a_Moires[ind_aM])

def get_list_fn(l):
    fn = ''
    for i in range(len(l)):
        fn += "{:.4f}".format(l[i])
        if i != len(l)-1:
            fn += '_'
    return fn

def get_fig_fn(e_cut,pars_grid,DFT,N,pars_V,a_M,pars_spread,machine):
    name_v = get_list_fn(pars_V)
    name_sp = get_list_fn(pars_spread[:2])
    return get_home_dn(machine)+'results/Figures/fig_'+"{:.4f}".format(e_cut)+'_'+pars_grid[0]+'_'+pars_spread[-1]+'_'+name_sp+'_'+"{:.2f}".format(pars_grid[1])+'_'+str(pars_grid[2])+'_'+str(DFT)+'_'+str(N)+'_'+name_v+'_'+"{:.1f}".format(a_M)+'.png'

def get_cut_fn(e_cut,pars_grid,DFT,N,pars_V,a_M,pars_spread,machine):
    name_v = get_list_fn(pars_V)
    name_sp = get_list_fn(pars_spread[:2])
    return get_home_dn(machine)+'results/data/E_cut_'+"{:.4f}".format(e_cut)+'_'+pars_grid[0]+'_'+pars_spread[-1]+'_'+name_sp+'_'+"{:.2f}".format(pars_grid[1])+'_'+str(pars_grid[2])+'_'+str(DFT)+'_'+str(N)+'_'+name_v+'_'+"{:.1f}".format(a_M)+'.npy'

def get_energies_fn(pars_grid,DFT,N,pars_V,a_M,machine):
    name_v = get_list_fn(pars_V)
    return get_home_dn(machine)+'results/data/energies_'+pars_grid[0]+'_'+"{:.2f}".format(pars_grid[1])+'_'+str(pars_grid[2])+'_'+str(DFT)+'_'+str(N)+'_'+name_v+'_'+"{:.1f}".format(a_M)+'.npy'

def get_weights_fn(pars_grid,DFT,N,pars_V,a_M,machine):
    name_v = get_list_fn(pars_V)
    return get_home_dn(machine)+'results/data/weights_'+pars_grid[0]+'_'+"{:.2f}".format(pars_grid[1])+'_'+str(pars_grid[2])+'_'+str(DFT)+'_'+str(N)+'_'+name_v+'_'+"{:.1f}".format(a_M)+'.npy'

def get_pars_mono_fn(TMD,machine,dft=False):
    get_dft = '_DFT' if dft else ''
    return get_home_dn(machine)+'inputs/pars_'+TMD+get_dft+'.npy'

def get_pars_interlayer_fn(machine,dft=False):
    get_dft = '_dft' if dft else ''
    return get_home_dn(machine)+'inputs/pars_interlayer'+get_dft+'.npy'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/last_lap/4_cem/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/4_cem/'
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

def get_Moire(a_M):     
    """Compute Moire reciprocal lattice vectors.

    """
    G_M = [0,4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
    G_M[0] = np.tensordot(R_z(-np.pi/3),G_M[1],1)
    return G_M

def R_z(t):
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R
