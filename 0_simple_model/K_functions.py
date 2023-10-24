import numpy as np
import matplotlib.pyplot as plt


def Hk_up(k,pars):
    m1,m2,m3,mu = pars
    return -k**2/2/m1 + k**4*m2 + k**6*m3 + mu

def V_g(g,pars):          #g is a integer from 0 to 5
    V,psi = pars
    return V*np.exp(1j*(-1)**g*psi)

def big_H(K_,N,pars_H,pars_V,G_M):
    n_cells = int(1+3*N*(N+1))
    H_up = np.zeros((n_cells,n_cells),dtype=complex)
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
            KKK = K_ + G_M[0]*lu[-1][0] + G_M[1]*lu[-1][1]
            K_p = np.linalg.norm(KKK)
            H_up[s,s] = Hk_up(K_p,pars_H)
    #Moirè
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
                H_up[s,nn] = V_g(g,pars_V)
    return H_up

def lorentzian_weight(k,e,*pars):
    spread_K,spread_E,weight_,K_,E_ = pars
    type_spread = 'gauss'
    if type_spread == 'lor':
        E2 = spread_E**2
        K2 = spread_K**2
        #f = 1000
        #weight_rescaled = weight_**2*f**2/(weight_**2*f**2+3*weight_*f+4)
        weight_rescaled = weight_**(1/2)
        return weight_rescaled/((k-K_)**2+K2)/((e-E_)**2+E2)
    elif type_spread == 'gauss':
        weight_rescaled = weight_**(1/2)
        s_e = spread_E
        s_k = spread_K
        return weight_rescaled*np.exp(-((k-K_)/s_k)**2)*np.exp(-((e-E_)/s_e)**2)

def path_BZ_GK(a_monolayer,pts_path,lim):
    path = []
    for i in range(pts_path):
        path.append(np.array([-lim+lim/pts_path*i,0]))
    return path
    G = 4*np.pi/np.sqrt(3)/a_monolayer*np.array([0,1])      #reciprocal lattice vector
    #
    K = np.tensordot(R_z(-np.pi/2),G,1)/np.sqrt(3)#
#    K = np.array([G[0]/3*2,0])                      #K-point
    Gamma = np.array([0,0])                                #Gamma
    Kp = np.tensordot(R_z(np.pi/3),K,1)     #K'-point
    path = []
    #G-K
    for i in range(pts_path):
        path.append(np.array([K[0]-lim+lim/pts_path*i,0]))
    return path

def image_difference(Pars, *args):
    V,phase,E_,K_ = Pars
    N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization = args
    fac_k = len_k//len(path)
    pars_V = (V,phase)
    n_cells = int(1+3*N*(N+1))
    res = np.zeros((len(path),n_cells))
    weight = np.zeros((len(path),n_cells))
    for i in range(len(path)):
        K = path[i] 
        H = big_H(K,N,pars_H,pars_V,G_M)
        res[i,:],evecs = np.linalg.eigh(H)
        for e in range(n_cells):
            weight[i,e] += np.abs(evecs[0,e])**2
    weight /= np.max(np.ravel(weight))
    # Plot single bands and weights
    if 0:
        #
        plt.figure()
        #plot all bands
        KKKK = np.linspace(K_list[0],K_list[-1],res.shape[0])
        for e in range(n_cells):
            plt.plot(KKKK,res[:,e],'k',linewidth=0.1)
        #plot all weigts
        for i in range(len(path)):
            for e in range(n_cells):
                if weight[i,e]>1e-3:
                    plt.scatter(KKKK[i],res[i,e],s=3*weight[i,e],color='b')
        if 1: #plot N=0 bands
            n_cells0 = 1
            res_0 = np.zeros((len(path),n_cells0))
            weight_0 = np.zeros((len(path),n_cells0))
            for i in range(len(path)):
                K = path[i]                                 #Considered K-point
                H = big_H(K,0,pars_H,pars_V,G_M)
                res_0[i,:],evecs = np.linalg.eigh(H)#,subset_by_index=[n_cells_below,n_cells-1])
            plt.plot(KKKK,res_0[:,0],'r',linewidth=0.5)
        plt.ylim(E_list[0],E_list[-1])       
        plt.xlim(KKKK[0],KKKK[-1])       
        plt.show()
        exit()
    #Lorentzian spread
    lor = np.zeros((len_k,len_e))
    for i in range(len(path)):
        for j in range(n_cells):
            if weight[i,j] > 1e-13:
                pars = (K_,E_,weight[i,j],K_list[i*fac_k],res[i,j])
                lor += lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
    #Transform lor to a png format in the range of white/black of the original picture
    max_lor = np.max(np.ravel(lor))
    min_lor = np.min(np.ravel(np.nonzero(lor)))
    whitest = 255#np.max(np.ravel(pic))     
    blackest = 0#np.min(np.ravel(pic))     
    norm_lor = np.zeros((len_k,len_e))
    for i in range(len_k):
        for j in range(len_e):
            norm_lor[i,j] = int((whitest-blackest)*(1-lor[i,j]/(max_lor-min_lor))+blackest)
    pic_lor = np.flip(norm_lor.T,axis=0)   #invert e-axis
    if 0: #png image
        from PIL import Image
        import os
        new_image = Image.fromarray(np.uint8(pic_lor))
        pars_name = "{:.4f}".format(V)+'_'+"{:.4f}".format(phase)+'_'+"{:.4f}".format(E_)+'_'+"{:.4f}".format(K_)
        new_imagename = "temp_image/"+pars_name+".png"
        new_image.save(new_imagename)
#        os.system("xdg-open "+new_imagename)
        #exit()
    if 0:#plot with pcolormesh on matplotlib
        X,Y = np.meshgrid(K_list,E_list)
        from matplotlib import cm
        from matplotlib.colors import LogNorm
        VMIN = lor[np.nonzero(lor)].min()
        VMAX = lor.max()
        plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=VMIN, vmax=VMAX))
        plt.show()
        exit()
    #Compute difference pixel by pixel of the two images
    minus = compute_difference(pic,pic_lor,len_e,len_k)
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
        return pic_lor

def compute_difference(pic,pic_lor,len_e,len_k):
    minus = np.absolute(np.ravel(pic[:,:len_k//2-120,0]-pic_lor[:,:len_k//2-120])).sum()        #the 120 comes from k2_fit_bands.py, line 53
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


def tqdm(n):
    return n

