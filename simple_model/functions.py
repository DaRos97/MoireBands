import numpy as np


def Hk_up(k,pars):
    a,b,c,m1,m2,mu = pars
    return -k**2/2/m1 + mu
def Hk_down(k,pars):
    a,b,c,m1,m2,mu = pars
    return -k**2/2/m2 -c + mu
def Hk_interlayer(k,pars):
    a,b,c,m1,m2,mu = pars
    return -a*(1-b*k**2)
def Hk(k,pars):     #just as a reference
    a,b,c,m1,m2,mu = pars
    return np.array([[-k**2/2/m1+mu,-a*(1-b*k**2)],[-a*(1-b*k**2),-k**2/2/m2-c+mu]])

def V_g(g,pars):          #g is a integer from 0 to 5
    V,psi = pars
    return V*np.exp(1j*(-1)**(g%2)*psi)

def big_H(K_,N,pars_H,pars_V,G_M):
    n_cells = int(1+3*N*(N+1))
    H_up = np.zeros((n_cells,n_cells),dtype=complex)
    H_down = np.zeros((n_cells,n_cells),dtype=complex)
    H_interlayer = np.zeros((n_cells,n_cells),dtype=complex)
    #
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
            K_p = np.linalg.norm(K_ + G_M[0]*lu[-1][0] + G_M[1]*lu[-1][1])
            #place the corresponding 22x22 Hamiltonian in its position
            H_up[s:s+1,s:s+1] = Hk_up(K_p,pars_H)
            H_down[s:s+1,s:s+1] = Hk_down(K_p,pars_H)
            H_interlayer[s:s+1,s:s+1] = Hk_interlayer(K_p,pars_H)
    #Moirè
    for n in range(0,N+1):      #Circles
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       #Indices inside the circle
            for i in m:
                ind_s = lu[s]
                ind_nn = (ind_s[0]+i[0],ind_s[1]+i[1])
                try:
                    nn = lu.index(ind_nn)
                except:
                    continue
                g = (m.index(i) + 2)%6
                H_up[s:(s+1),nn:(nn+1)] = V_g(g,pars_V)
                H_down[s:(s+1),nn:(nn+1)] = V_g(g,pars_V)
    #All together
    final_H = np.zeros((2*n_cells,2*n_cells),dtype=complex)
    final_H[:n_cells,:n_cells] = H_up
    final_H[n_cells:,n_cells:] = H_down
    final_H[n_cells:,:n_cells] = H_interlayer
    final_H[:n_cells,n_cells:] = np.conjugate(H_interlayer.T)
    return final_H

def pathBZ(path_name,a_monolayer,pts_ps):
    G = [4*np.pi/np.sqrt(3)/a_monolayer*np.array([0,1])]      
    for i in range(1,6):
        G.append(np.tensordot(R_z(np.pi/3*i),G[0],1))
    #
    K = np.array([G[-1][0]/3*2,0])                      #K-point
    Gamma = np.array([0,0])                                #Gamma
    K2 =    K/2                             #Half is denoted by a '2'
    K2_ =   - K2                            #Opposite wrt G is denoted by '_'
    M =     G[-1]/2                          #M-point
    M_ =     - M
    M2 =    M/2 
    M2_ =   - M2
    Kp =    np.tensordot(R_z(np.pi/3),K,1)     #K'-point
    Kp_ = - Kp
    Kp2 =   Kp/2
    Kp2_ =   -Kp2
    dic_names = {'G':Gamma,
                 'K':K/2,           ################
                 'M':M,
                 'm':M_,
                 'C':Kp/2,          ################
                 'c':Kp_, 
                 'Q':K2,
                 'q':K2_,
                 'N':M2,
                 'n':M2_,
                 'P':Kp2, 
                 'p':Kp2_,
                 }
    path = []
    for i in range(len([*path_name])-1):
        Pi = dic_names[path_name[i]]
        Pf = dic_names[path_name[i+1]]
        direction = Pf-Pi
        for i in range(pts_ps):
            path.append(Pi+direction*i/pts_ps)
    K_points = []
    for i in [*path_name]:
        K_points.append(dic_names[i])
    return path, K_points

def get_Moire(a_M):     #Compute Moire recipèrocal lattice vectors
    G_M = [4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
    for i in range(1,6):
        G_M.append(np.tensordot(R_z(np.pi/3*i),G_M[0],1))
    return G_M

def R_z(t):
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R

def lorentzian_weight(k,e,*pars):
    K2,E2,weight,K_,E_ = pars
    return abs(weight)/((k-K_)**2+K2)/((e-E_)**2+E2)

def tqdm(n):
    return n

def path_BZ_small(a_monolayer,pts_ps,lim):
    G = [4*np.pi/np.sqrt(3)/a_monolayer*np.array([0,1])]      
    for i in range(1,6):
        G.append(np.tensordot(R_z(np.pi/3*i),G[0],1))
    #
    K = np.array([G[-1][0]/3*2,0])                      #K-point
    Gamma = np.array([0,0])                                #Gamma
    Kp =    np.tensordot(R_z(np.pi/3),K,1)     #K'-point
    path = []
    #K-G
    m = K[1]/K[0]
    Kix = lim/np.sqrt(1+m**2) 
    Kiy = np.sqrt(lim**2-Kix**2)
    Ki = np.array([Kix,Kiy])
    Kf = Gamma
    direction = Kf-Ki
    for i in range(pts_ps):
        path.append(Ki+direction*i/pts_ps)
    #G-K'
    m = Kp[1]/Kp[0]
    Kix = lim/np.sqrt(1+m**2) 
    Kiy = np.sqrt(lim**2-Kix**2)
    Kf = np.array([Kix,Kiy])
    Ki = Gamma
    direction = Kf-Ki
    for i in range(pts_ps):
        path.append(Ki+direction*i/pts_ps)
    return path

def image_difference(Pars, *args):
    V,phase,E_,K_ = Pars
    N,pic,fig_E,fig_K,fac_grid_K,E_list,K_list,pars_H,G_M,path,minimization = args
    #
    pars_V = (V,phase)
    n_cells = int(1+3*N*(N+1))
    res = np.zeros((len(path),2*n_cells))
    weight = np.zeros((len(path),2*n_cells))
    for i in range(len(path)):
        K = path[i]                                 #Considered K-point
        H = big_H(K,N,pars_H,pars_V,G_M)
        res[i,:],evecs = np.linalg.eigh(H)#,subset_by_index=[n_cells_below,n_cells-1])
        for e in range(2*n_cells):
            for l in range(1):  #don't need the overlap with lower band, if not 1-->2
                weight[i,e] += np.abs(evecs[n_cells*l,e])**2       ################################
    K2 = K_**2
    E2 = E_**2
    lor = np.zeros((fig_K,fig_E))
    for i in range(len(path)):
        for j in range(2*n_cells):
            if weight[i,j] > 1e-3:
                pars = (K2,E2,weight[i,j],K_list[i*fac_grid_K],res[i,j])
                lor += lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
    #Transform lor to a png format
    max_lor = np.max(np.ravel(lor))
    for i in range(fig_K):
        for j in range(fig_E):
            lor[i,j] = int(256-256*lor[i,j]/max_lor)
    lor = np.uint8(np.flip(lor.T,axis=0))
    minus_image = (pic[:,:,0]-lor)
    minus = np.sum(np.ravel(minus_image))/(fig_E*fig_K)
    if minimization:
        return minus
    else:
        return lor

