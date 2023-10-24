import numpy as np
import matplotlib.pyplot as plt


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
    return V*np.exp(1j*(-1)**g*psi)

def big_H(K_,N,pars_H,pars_V,G_M,args_VI):
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
            KKK = K_ + G_M[0]*lu[-1][0] + G_M[1]*lu[-1][1]
            K_p = np.linalg.norm(KKK)
            H_up[s,s] = Hk_up(K_p,pars_H)
            H_down[s,s] = Hk_down(K_p,pars_H)
            H_interlayer[s,s] = Hk_interlayer(K_p,pars_H)
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
                H_down[s,nn] = V_g(g,pars_V)
                #H_interlayer[s,nn] = V_g(g,args_VI)
    #All together
    final_H = np.zeros((2*n_cells,2*n_cells),dtype=complex)
    final_H[:n_cells,:n_cells] = H_up
    final_H[n_cells:,n_cells:] = H_down
    final_H[n_cells:,:n_cells] = H_interlayer
    final_H[:n_cells,n_cells:] = np.conjugate(H_interlayer.T)
    return final_H

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
    return path

def image_difference(Pars, *args):
    V,phase,VI,phase_VI,spread_E,spread_K = Pars
    N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization = args
    fac_k = len_k//len(path)
    pars_V = (V,phase)
    args_VI = (VI,phase_VI)
    n_cells = int(1+3*N*(N+1))
    res = np.zeros((len(path),2*n_cells))
    weight = np.zeros((len(path),2*n_cells))
    for i in range(len(path)):
        K = path[i] 
        H = big_H(K,N,pars_H,pars_V,G_M,args_VI)
        res[i,:],evecs = np.linalg.eigh(H)
        for e in range(2*n_cells):
            for l in range(2):
                weight[i,e] += np.abs(evecs[n_cells*l,e])**2
    weight /= np.max(np.ravel(weight))
    if 0:   # Plot single bands and weights
        #
        K_space = np.linspace(-np.linalg.norm(path[0]),np.linalg.norm(path[-1]),len(path))
        plt.figure()
        #plot all bands
        for e in range(2*n_cells):
            plt.plot(K_space,res[:,e],'k',linewidth=0.1)
        #plot all weigts
        for i in range(len(path)):
            for e in range(2*n_cells):
                if weight[i,e]>1e-3:
                    if i > 3*len(path)//4 and i < 5*len(path)//6-4 and weight[i,e]>0.01 and res[i,e] > -1.2: #color some dots
                        #print(i,e,weight[i,e])
                        col='r'
                    else:
                        col='b'
                    plt.scatter(K_space[i],res[i,e],s=10*weight[i,e],color=col)
        if 1: #plot N=0 bands
            n_cells0 = 1
            res_0 = np.zeros((len(path),2*n_cells0))
            weight_0 = np.zeros((len(path),2*n_cells0))
            for i in range(len(path)):
                K = path[i]                                 #Considered K-point
                H = big_H(K,0,pars_H,pars_V,G_M,args_VI)
                res_0[i,:],evecs = np.linalg.eigh(H)#,subset_by_index=[n_cells_below,n_cells-1])
            plt.plot(K_space,res_0[:,0],'r',linewidth=0.5)
            plt.plot(K_space,res_0[:,1],'r',linewidth=0.5)
        plt.ylim(E_list[0],E_list[-1])       #-1.7,-0.5
        plt.xlim(K_space[0],K_space[-1])       #-0.5,0.5
        fignamee = str(N)+'_'+"{:.4f}".format(V).replace('.',',')+'_'+"{:.4f}".format(phase).replace('.',',')+'_'+"{:.4f}".format(VI).replace('.',',')+'_'+"{:.4f}".format(phase_VI).replace('.',',')+'.png'
        plt.savefig('/home/dario/Desktop/Figs_Moire/'+fignamee)
        #plt.show()
        exit()
    #Lorentzian spread
    lor = np.zeros((len_k,len_e))
    for i in range(len(path)):
        for j in range(2*n_cells):
            if weight[i,j] > 1e-3:
                pars = (spread_K,spread_E,weight[i,j],K_list[i*fac_k],res[i,j])
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
    if 0: #print png image
        from PIL import Image
        import os
        new_image = Image.fromarray(np.uint8(pic_lor))
        fignamee = str(N)+'_'+"{:.4f}".format(V).replace('.',',')+'_'+"{:.4f}".format(phase).replace('.',',')+'_'+"{:.4f}".format(VI).replace('.',',')+'_'+"{:.4f}".format(phase_VI).replace('.',',')+'.png'
        #pars_name = "{:.4f}".format(V)+'_'+"{:.4f}".format(phase)+'_'+"{:.4f}".format(E_)+'_'+"{:.4f}".format(K_)
        new_imagename = "/home/dario/Desktop/git/MoireBands/0_simple_model/temp_images/"+fignamee
        new_image.save(new_imagename)
        #os.system("xdg-open "+new_imagename)
        exit()
    if 0:#plot with pcolormesh on matplotlib
        X,Y = np.meshgrid(K_list,E_list)
        from matplotlib import cm
        from matplotlib.colors import LogNorm
        VMIN = 5e-3#lor[np.nonzero(lor)].min()
        VMAX = lor.max()
        for i in range(lor.shape[0]):
            for j in range(lor.shape[1]):
                if lor[i,j] < VMIN:
                    lor[i,j] = VMIN
        print(VMIN,VMAX)
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


def tqdm(n):
    return n

