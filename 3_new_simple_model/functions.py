import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

#A_M = 79.8#79.8  #Moirè lattice length (Angstrom)
a_mono = [3.32, 3.18]       #monolayer lattice lengths --> [WSe2, WS2] (Angstrom)
m_ = [[-1,1],[-1,0],[0,-1],[1,-1],[1,0],[0,1]]

def abs_diff(pars_H,*args):
    """Compute the absolute difference between the darkest points computed with pars_H and the fitted experimental data.
    
    Parameters
    ----------
    pars_H : np.ndarray
        Hamiltonian parameters: a,b,c,m1,m2,mu.
    args : tuple
        Fixed arguments:

    Returns
    -------
    float
        Absolute difference.
    """
    bounds_pic,pic,pts_layers,pars_V,pars_spread,N,path,G_M,removed_k = args
    other_args = (N,pic.shape[:2],path,G_M)
    picture = compute_image(pars_V,pars_H,pars_spread,bounds_pic,*other_args)
    DP = extract_darkest_points(picture,removed_k)
    if 1:   #plot picture with extracted points and border
        plot_step(picture,removed_k,DP,bounds_pic,pts_layers)
    #
    result = 0
    len_e, len_k = pic.shape[:2]
    for l in range(2):
        result += np.sum(np.absolute(pts_layers[l]-DP[l]))
    print(result)
    exit()
    return result

def plot_step(picture,removed_k,DP,bounds_pic,pts_layers):
    plt.figure(figsize=(20,20))
    len_e,len_k = picture.shape
    new_picture = np.zeros((len_e,len_k,4),dtype=int)
    col = (np.array([255,0,0,255]),np.array([0,255,0,255]),np.array([0,0,255,255]))
    bord = border(np.arange(len_k),len_e,len_k)
    for k in range(len_k):
        for e in range(len_e):
            new_picture[e,k] = np.array([picture[e,k],picture[e,k],picture[e,k],255])
        #Color border between the two bands
        new_picture[bord[k],k] = col[2]
    #Color extracted darkest points
    for l in range(2):
        for k in range(removed_k[l],len_k-removed_k[l]):
            new_picture[DP[l][k-removed_k[l]],k] = col[l]
        #Plot fitted values
        E_min, E_max, K_lim = bounds_pic
        K_lim_r = K_lim - removed_k[l]/len_k*2*K_lim
        K_space = np.linspace(removed_k[l],len_k-removed_k[l],len_k-2*removed_k[l])#np.linspace(-K_lim_r,K_lim_r,len_k-2*removed_k[l])
        plt.plot(K_space,pts_layers[l],'m',linewidth=0.5,label="exp fitted")
    plt.imshow(new_picture,cmap='gray')
    plt.legend()
    plt.show()
    return 0

def extract_darkest_points(picture,removed_k):
    """Compute pixel in energy axis at which the picture is darkest (no fitting with gaussian) for each band

    Parameters
    ----------
    picture : np.ndarray
        Picture.

    Returns
    -------
    np.ndarray
        2-Array of indexel in energy axis of blackest point for the 2 main bands (upper and lower).
    """
    len_e,len_k = picture.shape[:2]
    DP = []
    for l in range(2):
        DP.append(np.zeros(len_k-2*removed_k[l],dtype=int))
        for i in range(removed_k[l],len_k-removed_k[l]):
            bb = border(i,len_e,len_k)
            DP[l][i-removed_k[l]] = np.argmin(picture[:bb,i]) if l == 0 else bb + np.argmin(picture[bb:,i])
    return DP

def compute_image(pars_V,pars_H,pars_spread,bounds_pic,*args):
    """Compute image given a Moirè potential and Hamiltonian parameters.

    Parameters
    ----------
    pars_V : np.ndarray
        Moire potential parameters: V,phi.
    pars_H : np.ndarray
        Parameters of the Hamiltonian: a,b,c,m1,m2,mu.
    pars_spread : tuple
        Parameters of the spreading of weights: spread_K, spread_E and type_of_spread.
    args : tuple
        Fixed arguments:

    Returns
    -------
    np.ndarray
        Image of spreaded bands.
    """
    N, (len_e,len_k), path, G_M = args
    E_min,E_max,K_lim = bounds_pic
    fac_k = len_k//len(path)
    K_list = np.linspace(-K_lim,K_lim,len_k)
    E_list = np.linspace(E_min,E_max,len_e)
    n_cells = int(1+3*N*(N+1))
    Energies = np.zeros((len(path),2*n_cells))
    Weights = np.zeros((len(path),2*n_cells))
    LU = lu_table(N)
    for i in range(len(path)):
        Energies[i,:],evecs = np.linalg.eigh(big_H(path[i],N,pars_H,pars_V,G_M,LU))
        for l in range(2):
            for n in range(2*n_cells):
                Weights[i,n] += np.absolute(evecs[l*n_cells,n])**2
    ####NORMALIZE
    Weights /= np.max(np.ravel(Weights))
    if 0:
        plot_bands(path,N,Energies,Weights,pars_H,pars_V)
    ####
    #Lorentzian (or gaussian) spread
    lor = np.zeros((len_k,len_e))
    for i in range(len(path)):
        for n in range(2*n_cells):
            if 1:#Weights[i,n] > 1e-5:
                lor += weight_spreading(Weights[i,n],K_list[i*fac_k],Energies[i,n],K_list[:,None],E_list[None,:],pars_spread)
    #Transform lor to a png formati. in the range of white/black of the original picture
    max_lor = np.max(np.ravel(lor))
    min_lor = np.min(np.ravel(np.nonzero(lor)))
    whitest = 255
    blackest = 0     
    normalized_lor = np.zeros((len_k,len_e))
    for i in range(len_k):
        for j in range(len_e):
            normalized_lor[i,j] = int((whitest-blackest)*(1-lor[i,j]/(max_lor-min_lor))+blackest)
    picture = np.flip(normalized_lor.T,axis=0)   #invert e-axis to have the same structure
    if 0:
        plot_image(picture,bounds_pic)
    return picture

def main_bands(path,pars_H):
    """Compute energy bands of main BZ -> N=0.

    Parameters
    ----------
    pars_H : np.ndarray
        Parameters of the Hamiltonian: a,b,c,m1,m2,mu.

    Returns
    -------
    np.ndarray
        Energies of the 2 bands for all the k points.
    """
    energies_0 = np.zeros((len(path),2))
    for i in range(len(path)):
        K_i = path[i]
        energies_0[i,:],evecs = np.linalg.eigh(big_H(K_i,0,pars_H,(0,0),get_RLV(A_M),lu_table(0)))
    return energies_0

def plot_image(pictures,bounds_pic,m_b=False):
    """Plot picture.

    Parameters
    ----------
    pictures : np.ndarray
        Pictures to plot.
    """
    import matplotlib.pyplot as plt
    s_ = 15
    E_min,E_max,K_lim = bounds_pic
    plt.figure(figsize=(20,20))
    nn = len(pictures)
    col = 1 if nn == 1 else 2
    for i in range(nn):
        len_e, len_k = pictures[i].shape[:2]
        plt.subplot(nn//2+1,col,i+1)
        plt.imshow(pictures[i],cmap='gray')
        plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_lim),"0","{:.2f}".format(K_lim)])
        plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max),"{:.2f}".format((E_min+E_max)/2),"{:.2f}".format(E_min)])
        plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
        plt.ylabel("eV",size=s_)
        if m_b:
            K_space = np.linspace(0,len_k,len(path))
            Energies_0 = main_bands(path,pars_H)
            en_px = np.zeros((len(path),2))
            E_min_cut,E_max_cut = Energy_bounds
            for i in range(len(path)):
                en_px[i,:] = len_e*(E_max_cut-Energies_0[i,:])/(E_max_cut-E_min_cut)
            for d in range(2):
                plt.plot(K_space,en_px[:,d],'r',linewidth=0.5)
            plt.xlim(0,len_k)
            plt.ylim(len_e,0)
    plt.show()

def plot_bands(path,N,Energies,Weights,pars_H,pars_V):
    """Plot all bands and maybe the associated weights, highlighting the main band (of N=0).

    Parameters
    ----------
    path : list
        List of Kx,Ky values of the cut.
    N : int
        Number of considered circles of mini-BZ around the central one (N=0).
    Energies : np.ndarray
        Energies of all the bands.
    Weights : np.ndarray
        Weights of all the bands.
    """
    import matplotlib.pyplot as plt
    K_space = np.linspace(-np.linalg.norm(path[0]),np.linalg.norm(path[-1]),len(path))
    n_cells = int(1+3*N*(N+1))
    plt.figure()
    #plot all bands
    for e in range(2*n_cells):
        plt.plot(K_space,Energies[:,e],'k',linewidth=0.1)
    if 0:
        #plot all weigts
        for i in range(len(path)):
            for e in range(2*n_cells):
                if Weights[i,e]>1e-3:
                    plt.scatter(K_space[i],Energies[i,e],s=100*Weights[i,e],color='b')
    #N=0 bands
    Energies_0 = main_bands(path,pars_H)
    for d in range(2):
        plt.plot(K_space,Energies_0[:,d],'r',linewidth=0.5)
    plt.xlim(K_space[0],K_space[-1])       #-0.5,0.5
    plt.show()
    exit()

def big_H(momentum,N,pars_H,pars_V,G_M,LU):
    """Compute the multi-miniBZ Hamiltonian of two layers with Moirè potential and interlayer hopping.

    Parameters
    ----------
    K : np.ndarray
        Kx,Ky values.
    N : int
        Number of considered circles of mini-BZ around the central one (N=0).
    pars_H : np.ndarray
        Parameters of the Hamiltonian: a,b,c,m1,m2,mu.
    pars_V : np.ndarray
        Moire potential parameters: V,phi.
    G_M : np.ndarray
        Reciprocal Moirè lattice vectors.
    LU : list
        Look up table for idexes of mini-BZs in terms of G0 and G1.

    Returns
    -------
    np.ndarray
        Big ass Hamiltonian.
    """
    n_cells = int(1+3*N*(N+1))
    H_up = np.zeros((n_cells,n_cells),dtype=complex)
    H_down = np.zeros((n_cells,n_cells),dtype=complex)
    H_interlayer = np.zeros((n_cells,n_cells),dtype=complex)
    #
    for n in range(n_cells):      #circles go from 0 (central BZ) to N included
        KmBZ = momentum + G_M[0]*LU[n][0] + G_M[1]*LU[n][1]
        H_up[n,n] = Hk_ul(KmBZ,pars_H)
        H_down[n,n] = Hk_ll(KmBZ,pars_H)
        H_interlayer[n,n] = Hk_interlayer(KmBZ,pars_H)
    #Moirè
    for n in range(0,N+1):      #Circles
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       #Indices inside the circle
            ind_s = LU[s]
            for i in m_:
                ind_nn = (ind_s[0]+i[0],ind_s[1]+i[1])  #nn-> nearest neighbour
                try:
                    nn = LU.index(ind_nn)
                except:
                    continue
                g = m_.index(i)
                H_up[s,nn] = V_g(g,pars_V)
                H_down[s,nn] = V_g(g,pars_V)
    #All together
    final_H = np.zeros((2*n_cells,2*n_cells),dtype=complex)
    final_H[:n_cells,:n_cells] = H_up
    final_H[n_cells:,n_cells:] = H_down
    final_H[n_cells:,:n_cells] = H_interlayer
    final_H[:n_cells,n_cells:] = np.conjugate(H_interlayer.T)
    return final_H

def Hk_ul(K,pars):
    """Compute upper layer hamiltonian of single band model.

    Parameters
    ----------
    K : np.ndarray
        Kx,Ky values
    pars : np.ndarray
        Parameters of the Hamiltonian: a,b,c,m1,m2,mu.

    Returns
    -------
    float
        Upper layer energy.
    """
    k = np.linalg.norm(K)
    a,b,c,m1,m2,mu = pars
    return -k**2/2/m1 + mu
def Hk_ll(K,pars):
    """Compute lower layer hamiltonian of single band model.

    Parameters
    ----------
    K : np.ndarray
        Kx,Ky values
    pars : np.ndarray
        Parameters of the Hamiltonian: a,b,c,m1,m2,mu.

    Returns
    -------
    float
        Lower layer energy.
    """
    k = np.linalg.norm(K)
    a,b,c,m1,m2,mu = pars
    return -k**2/2/m2 -c + mu
def Hk_interlayer(K,pars):
    """Compute inter-layer hamiltonian of single band model.

    Parameters
    ----------
    K : np.ndarray
        Kx,Ky values
    pars : np.ndarray
        Parameters of the Hamiltonian: a,b,c,m1,m2,mu.

    Returns
    -------
    float
        Inter-layer energy.
    """
    k = np.linalg.norm(K)
    a,b,c,m1,m2,mu = pars
    return -a*(1-b*k**2)
def V_g(g,pars):
    """Compute Moire potential part of Hamiltonian.

    Parameters
    ----------
    g : int
        Moire reciprocal lattice vector index, which decides the sign of the phase.
    pars : np.ndarray
        Moire potential parameters: V,phi.

    Returns
    -------
    float
        Moirè potential energy.
    """
    V,psi = pars
    return V*np.exp(1j*(-1)**g*psi)
def lu_table(N):
    """Compute the look-up table containing the coordinates of each mini-BZ in terms of G0 and G1=(0,4pi/sqrt(3)/A_M).

    Parameters
    ----------
    N : int
        Number of considered circles of mini-BZ around the central one (N=0).

    Returns
    -------
    list
        Look up table in the form of list->each element is a 2-tuple containing the coefficients of G0 and G1.
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

def path_BZ_KGK(a_monolayer,pts_path,lim):
    """Compute cut in BZ.

    Parameters
    ----------
    a_monolayer : float
        Lattice length of reference (mono)layer.
    pts_path : int
       Points in momentum to consider. 
    lim : float
        Absolute value of K wrt Gamma to consider.

    Returns
    -------
    list
        List of K=(kx,Ky) (np.array) values in the path.
    """
    Ki = np.array([lim,0])
    Kf = -Ki
    path = []
    for i in range(pts_path):
        path.append(Ki+(Kf-Ki)/pts_path*i)
    return path

def get_RLV(a):
    """Compute first two reciprocal lattice vectors for a given lattice length.

    Parameters
    ----------
    a : float
        Lattice length.

    Returns
    -------
    list
        List of G0 and G1=(0,4pi/sqrt(3)/a.
    """
    G_a = [4*np.pi/np.sqrt(3)/a*np.array([0,1]),]    
    G_a.insert(0,np.tensordot(R_z(-np.pi/3),G_a[0],1))
    return G_a

def R_z(t):
    """Compute rotation matrix around z of angle t.

    Parameters
    ----------
    t : float
        Rotation angle.

    Returns
    -------
    np.ndarray
        2x2 rotation matrix.
    """
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R

def cut_image(bounds_pic,version,dirname,save=False):
    """Extracts the relevant window of parameters from experimental image.

    Parameters
    ----------
    bounds_pic : tuple
        Bounds of picture in physical parameters:
            -E_min : minimum energy in window.
            -E_max : maximum energy in window.
            -K_lim : range of K right and left of Gamma.
    version : string
        Defines which image to take from.
    dirname : string
        Name of directory where to take the experimental picture.
    save : bool
        Default False. Save or not the cut image in npy format.

    Returns
    -------
    np.ndarray
        Extracted picture.
    """
    E_min,E_max,K_lim = bounds_pic
    original_image = dirname + "KGK_WSe2onWS2_"+version+".png"
    K_i,K_f,E_i,E_f = (-1,1,0,-3.5) if version == 'v1' else (0,0,0,0)
    pic_0 = np.array(np.asarray(Image.open(original_image)))
    len_e, len_k, z = pic_0.shape
    #Empirically extracted for v1
    ki = 810
    kf = 2370
    ei = 85
    ef = 1908
    len_e = ef-ei
    len_k = kf-ki
    ind_Ei = ei + int((E_max-E_i)/(E_f-E_i)*len_e)
    ind_Ef = ei + int((E_min-E_i)/(E_f-E_i)*len_e)
    ind_Ki = ki + int(abs(K_i+K_lim)/(K_f-K_i)*len_k)
    ind_Kf = ind_Ki + int(2*K_lim/(K_f-K_i)*len_k)
    #get Energy of relevant window
    pic = pic_0[ind_Ei:ind_Ef,ind_Ki:ind_Kf]
    if save:
        np.save(compute_picture_filename(version,bounds_pic,dirname),pic)
    return pic

def compute_picture_filename(version,bounds_pic,dirname):
    """Computes name of picture given cut parameters.

    Parameters
    ----------
    version : string
        Defines which image to take from.
    bounds_pic : tuple
        Bounds of picture in physical parameters:
            -E_min : minimum energy in window.
            -E_max : maximum energy in window.
            -K_lim : range of K right and left of Gamma.
    dirname : string
        Name of directory where to take the experimental picture.

    Returns
    -------
    string
        Picture name.
    """
    E_min,E_max,K_lim = bounds_pic
    return dirname + "cut_KGK_"+version+"_"+"{:.2f}".format(E_min)+"_"+"{:.2f}".format(E_max)+"_"+"{:.2f}".format(K_lim)+".npy"

def compute_pts_filename(layer,version,bounds_pic,dirname,removed_k):
    """Computes name of picture given cut parameters.

    Parameters
    ----------
    layer : string
        Which layer: ul or ll.
    version : string
        Defines which image to take from.
    bounds_pic : tuple
        Bounds of picture in physical parameters:
            -E_min : minimum energy in window.
            -E_max : maximum energy in window.
            -K_lim : range of K right and left of Gamma.
    dirname : string
        Name of directory of file.

    Returns
    -------
    string
        Picture name.
    """
    E_min,E_max,K_lim = bounds_pic
    return dirname + "pts_"+layer+"_"+version+"_"+"{:.2f}".format(E_min)+"_"+"{:.2f}".format(E_max)+"_"+"{:.2f}".format(K_lim)+"{:.1f}".format(removed_k)+".npy"

def compute_Hopt_filename(N,pars_V,pars_spread,version,bounds_pic,dirname,removed_k):
    """Computes name of picture given cut parameters.

    Parameters
    ----------
    layer : string
        Which layer: ul or ll.
    version : string
        Defines which image to take from.
    bounds_pic : tuple
        Bounds of picture in physical parameters:
            -E_min : minimum energy in window.
            -E_max : maximum energy in window.
            -K_lim : range of K right and left of Gamma.
    dirname : string
        Name of directory of file.

    Returns
    -------
    string
        Picture name.
    """
    E_min,E_max,K_lim = bounds_pic
    return dirname + "Hopt_"+str(N)+"_"+"{:.5f}".format(pars_V[0])+"_"+"{:.5f}".format(pars_V[1])+"_"+"{:.5f}".format(pars_spread[0])+"_"+"{:.5f}".format(pars_spread[1])+"_"+pars_spread[2]+"_"+version+"_"+"{:.2f}".format(E_min)+"_"+"{:.2f}".format(E_max)+"_"+"{:.2f}".format(K_lim)+"_"+"{:.1f}".format(removed_k[0])+"_"+"{:.1f}".format(removed_k[1])+".npy"

def compute_bopt_filename(dirname,args_minimization):
    """Computes name of picture given cut parameters.

    Parameters
    ----------
    layer : string
        Which layer: ul or ll.
    version : string
        Defines which image to take from.
    bounds_pic : tuple
        Bounds of picture in physical parameters:
            -E_min : minimum energy in window.
            -E_max : maximum energy in window.
            -K_lim : range of K right and left of Gamma.
    dirname : string
        Name of directory of file.

    Returns
    -------
    string
        Picture name.
    """
    N,pars_spread,phi,Hopt,bounds_pic,path,pic,bool_min = args_minimization
    factor_k =  pic.shape[1]//len(path)
    return dirname + "bopt_"+str(N)+'_'+"{:.4f}".format(phi)+'_'+str(factor_k)+"_"+"{:.5f}".format(pars_spread[0])+"_"+pars_spread[2]+".npy"
def compute_bopt_figname(dirname,args_minimization):
    """Computes name of picture given cut parameters.

    Parameters
    ----------
    layer : string
        Which layer: ul or ll.
    version : string
        Defines which image to take from.
    bounds_pic : tuple
        Bounds of picture in physical parameters:
            -E_min : minimum energy in window.
            -E_max : maximum energy in window.
            -K_lim : range of K right and left of Gamma.
    dirname : string
        Name of directory of file.

    Returns
    -------
    string
        Picture name.
    """
    N,pars_spread,phi,Hopt,bounds_pic,path,pic,bool_min = args_minimization
    factor_k =  pic.shape[1]//len(path)
    return dirname + "bopt_fig_"+str(N)+'_'+"{:.4f}".format(phi)+'_'+str(factor_k)+"_"+"{:.5f}".format(pars_spread[0])+"_"+pars_spread[2]+".npy"

def compute_pts(picture,bounds_pic,version,dirname,removed_k,save=False):
    """Computes optimal parameters fitting the experimental image, using a polinomial.

    Parameters
    ----------
    picture : np.ndarray
        Picture to fit.
    bounds_pic : tuple
        Bounds of picture in physical parameters:
            -E_min : minimum energy in window.
            -E_max : maximum energy in window.
            -K_lim : range of K right and left of Gamma.
    dirname : string
        Name of directory where to save the result.

    Returns
    -------
    string
        Picture name.
    """
    E_min,E_max,K_lim = bounds_pic
    len_e, len_k, z = picture.shape
    #
    red = np.array([255,0,0,255])
    green = np.array([0,255,0,255])
    blue = np.array([0,0,255,255])
    if 0:#print border between the two bands
        for x in range(len_k):
            bb = border(x,len_e,len_k)
            picture[bb,x] = green
        plt.imshow(picture,cmap='gray')
        plt.show()
        exit()
    #Extract Darkest points
    data_ul = np.zeros(len_k-2*removed_k[0],dtype=int)
    data_ll = np.zeros(len_k-2*removed_k[1],dtype=int)
    for x in range(removed_k[0],len_k-removed_k[0]):
        bb = border(x,len_e,len_k)
        col_up = picture[:bb,x,0]
        d_up = find_max(col_up)
        picture[d_up,x,:] = red
        data_ul[x-removed_k[0]] = int(len_e-d_up)
    for x in range(removed_k[1],len_k-removed_k[1]):
        bb = border(x,len_e,len_k)
        col_low = picture[bb:,x,0]
        d_low = find_max(col_low)
        picture[bb+d_low,x,:] = blue
        data_ll[x-removed_k[1]] = int(len_e-(bb+d_low))
    if 0:   #plot taken points
        plot_image(picture,bounds_pic)
    #Fit poly on points of ul
    K_lim_r = K_lim - removed_k[0]/len_k*2*K_lim
    popt_ul,pcov_ul = curve_fit(
            poly,
            np.linspace(-K_lim_r,K_lim_r,len_k-2*removed_k[0]),
            np.linspace(E_min,E_max,len_e)[data_ul],
            p0=(-0.9,0,-1,0,-0.1),
            bounds=([-10,-10,-10,-10,-10],[10,10,10,10,10]),
            )
    pts_ul = len_e*(E_max-poly(np.linspace(-K_lim_r,K_lim_r,len_k-2*removed_k[0]),*popt_ul))/(E_max-E_min)
    #Fit poly on points of ll
    K_lim_r = K_lim - removed_k[1]/len_k*2*K_lim
    popt_ll,pcov_ll = curve_fit(
            poly,
            np.linspace(-K_lim_r,K_lim_r,len_k-2*removed_k[1]),
            np.linspace(E_min,E_max,len_e)[data_ll],
            p0=(-0.9,0,-1,0,-0.1),
            bounds=([-10,-10,-10,-10,-10],[10,10,10,10,10]),
            )
    pts_ll = len_e*(E_max-poly(np.linspace(-K_lim_r,K_lim_r,len_k-2*removed_k[1]),*popt_ll))/(E_max-E_min)
    if save:
        np.save(compute_pts_filename('ul',version,bounds_pic,dirname,removed_k[0]),pts_ul)
        np.save(compute_pts_filename('ll',version,bounds_pic,dirname,removed_k[1]),pts_ll)
    if 0:   #plot ll
        plt.figure(figsize=(20,20))
        plt.imshow(picture)
        new_k = np.arange(removed_k[1],len_k-removed_k[1])
        plt.plot(new_k,pts_ll,'r')
        plt.show()
        exit()
    return pts_ul, pts_ll

def poly(x,a,b,c,d,e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4
def border(x,len_e,len_k):
    return len_e//2+(x-len_k//2)**2//600 #400
def inv_gauss(x,a,b,x0,s):
    return -(a*np.exp(-((x-x0)/s)**2)+b)
def find_max(col):
    med = np.argmin(col)
    domain = 10
    in_ = med-domain if med-domain > 0 else 0
    fin_ = med+domain if med+domain < len(col) else -1
    new_arr = col[in_:fin_]
    P0 = [np.max(new_arr)-np.min(new_arr),-np.max(new_arr),np.argmin(new_arr),50]
    try:
        popt,pcov = curve_fit(
            inv_gauss, 
            np.arange(len(new_arr)), new_arr,
            p0 = P0,
            )
        return in_+int(popt[2]) if abs(in_+int(popt[2]))<len(col) else med
    except:
        return med

def plot_final(pic1, pic2, bounds_pic,filename,pars_V,A_M):
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    s_ = 15
    E_min,E_max,K_lim = bounds_pic
    fig = plt.figure(figsize=(20,10))
    len_e, len_k = pic1.shape[:2]
    plt.suptitle("A_M: "+"{:.2f}".format(A_M)+", (V,phi)=("+"{:.4f}".format(pars_V[0])+","+"{:.4f}".format(pars_V[1])+")")
    plt.subplot(1,2,1)
    #plt.imshow(pic1,cmap='gray')
    plt.imshow(pic1[:,:,0],cmap=cm.gray)
    plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_lim),"0","{:.2f}".format(K_lim)])
    plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max),"{:.2f}".format((E_min+E_max)/2),"{:.2f}".format(E_min)])
    plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
    plt.ylabel("eV",size=s_)
    plt.subplot(1,2,2)
    #plt.imshow(pic2,cmap = cm.gray_r, norm=LogNorm(vmin=1e-3, vmax=1e4))
    plt.imshow(pic2,cmap=cm.gray)
    plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_lim),"0","{:.2f}".format(K_lim)])
    plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max),"{:.2f}".format((E_min+E_max)/2),"{:.2f}".format(E_min)])
    plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
#    plt.ylabel("eV",size=s_)
    if 0:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def difference_bopt(pars,*args):
    A_M, V, spread_E = pars
    N, in_pars_spread, phi, Hopt, bounds_pic, path, pic, minimization = args
    pars_spread = (in_pars_spread[0],spread_E,in_pars_spread[2])
    pars_V = (V,phi)
    args_pic = (N,pic.shape[:2],path,get_RLV(A_M))
    picture = compute_image(pars_V,Hopt,pars_spread,bounds_pic,*args_pic)
    if minimization:
        result = np.sum(np.absolute(picture-pic[:,:,0]))
        print(pars,result)
        plot_image((picture,pic),bounds_pic)
        return result
    else:
        return picture


    
















