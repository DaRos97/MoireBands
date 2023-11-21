import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

offset_3B = {'WSe2':-1, 'WS2':-0.7}
def Hk_three(k,pars,material):
    k_x,k_y = k				#momentum
    lattice_constant,z_xx,e_1,e_2,t_0,t_1,t_2,t_11,t_12,t_22,r_0,r_1,r_2,r_11,r_12,u_0,u_1,u_2,u_11,u_12,u_22,lamb = pars[material]
    a = k_x*lattice_constant/2              #alpha
    b = k_y*lattice_constant*np.sqrt(3)/2   #beta
    V_0 = (e_1 + 2*t_0*(2*np.cos(a)*np.cos(b)+np.cos(2*a)) 
                + 2*r_0*(2*np.cos(3*a)*np.cos(b)+np.cos(2*b))
           +2*u_0*(2*np.cos(2*a)*np.cos(2*b)+np.cos(4*a))
           )
    V_1 = complex(-2*np.sqrt(3)*t_2*np.sin(a)*np.sin(b)
                  +2*(r_1+r_2)*np.sin(3*a)*np.sin(b)
                  -2*np.sqrt(3)*u_2*np.sin(2*a)*np.sin(2*b),
                  2*t_1*np.sin(a)*(2*np.cos(a)+np.cos(b))
                  +2*(r_1-r_2)*np.sin(3*a)*np.cos(b)
                  +2*u_1*np.sin(2*a)*(2*np.cos(2*a)+np.cos(2*b))
                )
    V_2 = complex(2*t_2*(np.cos(2*a)-np.cos(a)*np.cos(b))
                  -2/np.sqrt(3)*(r_1+r_2)*(np.cos(3*a)*np.cos(b)-np.cos(2*b))
                  +2*u_2*(np.cos(4*a)-np.cos(2*a)*np.cos(2*b)),
                  2*np.sqrt(3)*t_1*np.cos(a)*np.sin(b)
                  +2/np.sqrt(3)*np.sin(b)*(r_1-r_2)*(np.cos(3*a)+2*np.cos(b))
                  +2*np.sqrt(3)*u_1*np.cos(2*a)*np.sin(2*b)
                )
    V_11 = (e_2 + (t_11+3*t_22)*np.cos(a)*np.cos(b) + 2*t_11*np.cos(2*a)
            +4*r_11*np.cos(3*a)*np.cos(b) + 2*(r_11+np.sqrt(3)*r_12)*np.cos(2*b)
            +(u_11+3*u_22)*np.cos(2*a)*np.cos(2*b) + 2*u_11*np.cos(4*a)
            )
    V_12 = complex(np.sqrt(3)*(t_22-t_11)*np.sin(a)*np.sin(b) + 4*r_12*np.sin(3*a)*np.sin(b)
                   +np.sqrt(3)*(u_22-u_11)*np.sin(2*a)*np.sin(2*b),
                   4*t_12*np.sin(a)*(np.cos(a)-np.cos(b))
                   +4*u_12*np.sin(2*a)*(np.cos(2*a)-np.cos(2*b))
                )
    V_22 = (e_2 + (3*t_11+t_22)*np.cos(a)*np.cos(b) + 2*t_22*np.cos(2*a)
            +2*r_11*(2*np.cos(3*a)*np.cos(b)+np.cos(2*b))
            +2/np.sqrt(3)*r_12*(4*np.cos(3*a)*np.cos(b)-np.cos(2*b))
            +(3*u_11+u_22)*np.cos(2*a)*np.cos(2*b) + 2*u_22*np.cos(4*a)
            )
    H_0 = np.array([[V_0,V_1,V_2],
                    [np.conjugate(V_1),V_11,V_12],
                    [np.conjugate(V_2),np.conjugate(V_12),V_22]])
    L_z = np.zeros((3,3),dtype = complex)
    L_z[1,2] = 2*1j;    L_z[2,1] = -2*1j
    Sig_3 = np.zeros((2,2),dtype=complex)
    Sig_3[0,0] = 1; Sig_3[1,1] = -1
    Hp = lamb/2*np.kron(Sig_3,L_z)
    Id = np.identity(2)
    H_final = np.kron(Id,H_0) + Hp
    return H_final  +   offset_3B[material]*np.identity(6)

def Hk_0(k,pars):
    return np.zeros((6,6),dtype=complex)
def Hk_up(K,pars,t):
    k = np.linalg.norm(K)
    a,b,c,m1,m2,mu = pars
    return -k**2/2/m1 + mu
def Hk_down(K,pars,t):
    k = np.linalg.norm(K)
    a,b,c,m1,m2,mu = pars
    return -k**2/2/m2 -c + mu
def Hk_interlayer(K,pars):
    k = np.linalg.norm(K)
    a,b,c,m1,m2,mu = pars
    return -a*(1-b*k**2)
def Hk(k,pars):     #just as a reference
    a,b,c,m1,m2,mu = pars
    return np.array([[-k**2/2/m1+mu,-a*(1-b*k**2)],[-a*(1-b*k**2),-k**2/2/m2-c+mu]])

def V_g(g,pars,MM):          #g is a integer from 0 to 5
    #Not considering G and K different in Moire type for different orbitals -> ok only close to Gamma
    V,psi = pars
    return np.identity(MM,dtype=complex)*V*np.exp(1j*(-1)**g*psi)

def lu_table(N,G_M):
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
    if 0:   #Plot mini-BZ k points
        plt.figure()
        for n in range(n_cells):
            r = lu[n][0]*G_M[0] + lu[n][1]*G_M[1]
            plt.scatter(r[0],r[1],label=str(n))
        plt.legend()
        plt.gca().set_aspect('equal')
        plt.show()
        exit()
    return lu

def big_H(K_,N,pars_H,pars_V,fun_H,MM,G_M,lu):
    n_cells = int(1+3*N*(N+1))*MM
    H_up = np.zeros((n_cells,n_cells),dtype=complex)
    H_down = np.zeros((n_cells,n_cells),dtype=complex)
    H_interlayer = np.zeros((n_cells,n_cells),dtype=complex)
    #
    for n in range(n_cells//MM):      #circles go from 0 (central BZ) to N included
        KKK = K_ + G_M[0]*lu[n][0] + G_M[1]*lu[n][1]
        #K_p = np.linalg.norm(KKK)
        H_up[n*MM:n*MM+MM,n*MM:n*MM+MM] = fun_H[0](KKK,pars_H,'WS2')
        H_down[n*MM:n*MM+MM,n*MM:n*MM+MM] = fun_H[1](KKK,pars_H,'WSe2')
        H_interlayer[n*MM:n*MM+MM,n*MM:n*MM+MM] = fun_H[2](KKK,pars_H)
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
                H_up[s*MM:s*MM+MM,nn*MM:nn*MM+MM] = V_g(g,pars_V,MM)
                H_down[s*MM:s*MM+MM,nn*MM:nn*MM+MM] = V_g(g,pars_V,MM)
    #All together
    final_H = np.zeros((2*n_cells,2*n_cells),dtype=complex)
    final_H[:n_cells,:n_cells] = H_up
    final_H[n_cells:,n_cells:] = H_down
    final_H[n_cells:,:n_cells] = H_interlayer
    final_H[:n_cells,n_cells:] = np.conjugate(H_interlayer.T)
    return final_H

def lorentzian_weight(k,e,*pars):
    spread_K,spread_E,weight_,K_,E_ = pars
    type_spread = 'lor'
    if type_spread == 'lor':
        E2 = spread_E**2
        K2 = spread_K**2
        #f = 1000
        #weight_rescaled = weight_**2*f**2/(weight_**2*f**2+3*weight_*f+4)
        weight_rescaled = weight_#**(1/2)
        return weight_rescaled/((k-K_)**2+K2)/((e-E_)**2+E2)
    elif type_spread == 'gauss':
        weight_rescaled = weight_#**(0.5)
        s_e = spread_E
        s_k = spread_K
        return weight_rescaled*np.exp(-((k-K_)/s_k)**2)*np.exp(-((e-E_)/s_e)**2)

def path_BZ_KGK(a_monolayer,pts_path,lim):
    G = 4*np.pi/np.sqrt(3)/a_monolayer*np.array([0,1])      #reciprocal lattice vector
    #
    K = np.tensordot(R_z(-np.pi/2),G,1)/np.sqrt(3)#
    Kp = -K
    Gamma = np.array([0,0])                                #Gamma
    M = G/2 
    Ki = K
    Ki = Ki/np.linalg.norm(Ki)*lim
    Kf = -Ki
    path = []
    for i in range(pts_path):
        path.append(Ki+(Kf-Ki)/pts_path*i)
    if 0:   #plot path in BZ
        plt.figure()
        for i in range(len(path)):
            plt.scatter(path[i][0],path[i][1],color='k',marker='*')
        plt.gca().set_aspect('equal')
        plt.show()
        exit()
    return path

def image_difference(Pars, *args):
    V,phase,spread_E,spread_K = Pars
    N,pic,len_e,len_k,E_list,K_list,pars_H,bands_type,G_M,path,minimization = args
    if bands_type == 1:
        fun_H = (Hk_up,Hk_down,Hk_interlayer) 
    elif bands_type == 3:
        fun_H = (Hk_three,Hk_three,Hk_0) 
    elif bands_type == 11:
        fun_H = (Hk_eleven,Hk_eleven,Hk_0) 
    MM = 1 if bands_type == 1 else 2*bands_type
    #
    fac_k = len_k//len(path)
    pars_V = (V,phase)
    n_cells = int(1+3*N*(N+1))*MM        #possible to reduce it for 3 and 11 -> consider only some valence bands
    res = np.zeros((len(path),2*n_cells))
    weight = np.zeros((len(path),2*n_cells))
    lu = lu_table(N,G_M)
    for i in tqdm(range(len(path))):
        K = path[i] 
        H = big_H(K,N,pars_H,pars_V,fun_H,MM,G_M,lu)
        res[i,:],evecs = np.linalg.eigh(H)
#        evecs = np.flip(evecs,axis=1)
        for l in range(2):
            for n in range(2*n_cells):
                weight[i,n] += np.absolute(evecs[l*n_cells,n])**2
#        for n in range(n_cells,2*n_cells):
#            weight[i,n] = np.absolute(evecs[0,n])**2
        continue
        for l in range(2):
            for n in range(n_cells//MM):
#                k_mBZ = K + lu[n][0]*G_M[0] + lu[n][1]*G_M[1]
                for d in range(MM):
                    weight[i,l*n_cells+n*MM+d] += np.abs(evecs[n_cells*l,n_cells*l+n*MM+d])**2  #  *k_mBZ[0]**2
    ####NORMALIZE
    weight /= np.max(np.ravel(weight))
    ####
    if 0:   # Plot single bands and weights
        #
        K_space = np.linspace(-np.linalg.norm(path[0]),np.linalg.norm(path[-1]),len(path))
        plt.figure()
        #plot all bands
        for e in range(2*n_cells):
            plt.plot(K_space,res[:,e],'k',linewidth=0.1)
        #plot all weigts
        for i in range(len(path)):
            for e in range(n_cells,2*n_cells):
                if weight[i,e]>1e-3:
                    if i > 3*len(path)//4 and i < 5*len(path)//6-4 and weight[i,e]>0.01 and res[i,e] > -1.2: #color some dots
                        #print(i,e,weight[i,e])
                        col='r'
                    else:
                        col='b'
                    plt.scatter(K_space[i],res[i,e],s=100*weight[i,e],color=col)
        if 1: #plot N=0 bands
            n_cells0 = MM
            res_0 = np.zeros((len(path),2*n_cells0))
            weight_0 = np.zeros((len(path),2*n_cells0))
            for i in range(len(path)):
                K = path[i]                                 #Considered K-point
                H = big_H(K,0,pars_H,pars_V,fun_H,MM,G_M,lu_table(0,G_M))
                res_0[i,:],evecs = np.linalg.eigh(H)#,subset_by_index=[n_cells_below,n_cells-1])
            for d in range(2*MM):
                plt.plot(K_space,res_0[:,d],'r',linewidth=0.5)
#        plt.ylim(E_list[0],E_list[-1])       #-1.7,-0.5
        plt.xlim(K_space[0],K_space[-1])       #-0.5,0.5
        fignamee = str(N)+'_'+"{:.4f}".format(V).replace('.',',')+'_'+"{:.4f}".format(phase).replace('.',',')+'.png'
        #plt.savefig('/home/dario/Desktop/Figs_Moire/'+fignamee)
        plt.show()
        exit()
    #Lorentzian spread
    lor = np.zeros((len_k,len_e))
    for i in range(len(path)):
        for n in range(2*n_cells):
            if weight[i,n] > 1e-5:
                pars = (spread_K,spread_E,weight[i,n],K_list[i*fac_k],res[i,n])
                lor += lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
    #Transform lor to a png format in the range of white/black of the original picture
    max_lor = np.max(np.ravel(lor))
    min_lor = np.min(np.ravel(np.nonzero(lor)))
    whitest = np.max(np.ravel(pic))     
    blackest = np.min(np.ravel(pic))     
    norm_lor = np.zeros((len_k,len_e))
    for i in range(len_k):
        for j in range(len_e):
            norm_lor[i,j] = int((whitest-blackest)*(1-lor[i,j]/(max_lor-min_lor))+blackest)
    pic_lor = np.flip(norm_lor.T,axis=0)   #invert e-axis
    if 1:
        import matplotlib.pyplot as plt
        s_ = 15
        plt.figure(figsize=(10,9))
        plt.imshow(pic_lor,cmap='gray')
        #plt.text(len_k-260,30,"V="+"{:.1f}".format(V*1000)+" meV, $\phi$="+"{:.2f}".format(phase)+" rad",size = s_)
        #plt.xticks([0,len_k//2,len_k],["-0.5","0","0.5"])
        plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
        plt.yticks([0,len_e//2,len_e],["-0.55","-0.9","-1.25"])
        plt.ylabel("eV",size=s_)
        if 1:
            n_cells0 = MM
            K_space = np.linspace(0,len_k,len(path))
            fac = len_k//len(path)
            res_0 = np.zeros((len(path),2*n_cells0))
            weight_0 = np.zeros((len(path),2*n_cells0))
            en_px = np.zeros((len(path),2))
            E_min_cut = -2.2
            E_max_cut = -0.9#-0.95#-0.5
            for i in range(len(path)):
                K = path[i]                                 #Considered K-point
                H = big_H(K,0,pars_H,pars_V,fun_H,MM,G_M,lu_table(0,G_M))
                res_0[i,:],evecs = np.linalg.eigh(H)#,subset_by_index=[n_cells_below,n_cells-1])
                en_px[i,:] = len_e*(E_max_cut-res_0[i,:])/(E_max_cut-E_min_cut)
            for d in range(2*MM):
                plt.plot(K_space,en_px[:,d],'r',linewidth=0.5)
            plt.xlim(0,len_k)
            plt.ylim(len_e,0)
        plt.show()
        exit()
    #Compute difference pixel by pixel of the two images
    minus = 0#compute_difference(pic,pic_lor,len_e,len_k)
    #
    if minimization:
        if 0:   #interacting minimization
            print(Pars)
            print("Minus: ",minus)
            a = input("plot? (y/N)")
            if a=='y': #print png image
                from PIL import Image
                import os
                new_image = Image.fromarray(np.uint8(pic_lor))
                new_imagename = "temp.png"
                new_image.save(new_imagename)
                os.system("xdg-open "+new_imagename)
        return minus
    else:
        return pic_lor, minus

def compute_difference(pic,pic_lor,len_e,len_k):
    minus = np.absolute(np.ravel(pic[:,:len_k//2,0]-pic_lor[:,:len_k//2])).sum()
    return minus/pic_lor.shape[0]/pic_lor.shape[1]



def get_Moire(a_M):     #Compute Moire recipèrocal lattice vectors
    G_M = [0,4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
    G_M[0] = np.tensordot(R_z(-np.pi/3),G_M[1],1)
    if 0:
        plt.figure()
        plt.scatter(0,0)
        plt.scatter(G_M[0][0],G_M[0][1],color='k')
        plt.scatter(G_M[1][0],G_M[1][1],color='r')
        plt.gca().set_aspect('equal')
        plt.show()
        exit()
    return G_M

def R_z(t):
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R


def tqdm_i(n):
    return n


def gridBZ(grid_pars,a_monolayer):
    K_center,dist_kx,dist_ky,pts_per_direction = grid_pars
    #K_center: string with name of central point of the grid
    #dist_k*: float of distance from central point of furthest point in each direction *
    #pts_per_direction: array of 2 floats with TOTAL number of steps in the two directions -> better if odd so central point is included
    G = [4*np.pi/np.sqrt(3)/a_monolayer*np.array([0,1]),]      
    for i in range(1,6):
        G.append(np.tensordot(R_z(np.pi/3*i),G[0],1))
    K = np.array([G[-1][0]/3*2,0])                      #K-point
    Gamma = np.array([0,0])                             #Gamma
    Kp =    np.tensordot(R_z(np.pi/3),K,1)              #K'-point
    dic_symm_pts = {'G':Gamma,'K':K,'C':Kp}
    #
    grid = np.zeros((pts_per_direction[0],pts_per_direction[1],2))
    KKK = dic_symm_pts[K_center]
    for x in range(-pts_per_direction[0]//2,pts_per_direction[0]//2+1):
        for y in range(-pts_per_direction[1]//2,pts_per_direction[1]//2+1):
            K_pt_x = KKK[0] + 2*dist_kx*x/pts_per_direction[0]
            K_pt_y = KKK[1] + 2*dist_ky*y/pts_per_direction[1]
            grid[x+pts_per_direction[0]//2,y+pts_per_direction[1]//2,0] = K_pt_x
            grid[x+pts_per_direction[0]//2,y+pts_per_direction[1]//2,1] = K_pt_y
    return grid

def spread_lor(K,sp_kx,sp_ky,kx_list,ky_list):
    return 1/((kx_list[:,None]-K[0])**2+(ky_list[None,:]-K[1])**2+sp_kx**2)**(3/2)
def spread_gauss(K,sp_kx,sp_ky,kx_list,ky_list):
    return np.exp(-((K[0]-kx_list[:,None])/sp_kx)**2)*np.exp(-((K[1]-ky_list[None,:])/sp_ky)**2)

spread_fun_dic = {'lor':spread_lor, 'gauss':spread_gauss}
