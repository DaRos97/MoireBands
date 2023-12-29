import numpy as np


def energy_mono(k_,pars):
    """Computes the energy of the single bands around K (taken to be the center of coordinates).
    The formula interpolates between the slopes of the parabola going towards Gamma and M.

    Parameters
    ----------
    k_: 2-tuple
        momentum.
    pars: 3-tuple
        m_g, m_m and mu parameters, needed for the energy.

    Returns
    -------
    float
        Energy
    """
    m_g,m_m,mu = pars
    k_x,k_y = k_
    theta = 0 if (k_x==0 and k_y==0) else np.arctan2(k_y,k_x)
    return mu - (k_x**2+k_y**2)/2/m_m*(np.cos(3/2*theta)**2+m_m/m_g*np.sin(3/2*theta)**2)

def energy_inter(k_,pars):
    """Computes the energy of the single bands around K (taken to be the center of coordinates).
    The formula interpolates between the slopes of the parabola going towards Gamma and M.

    Parameters
    ----------
    k_: 2-tuple
        momentum.
    pars: 3-tuple
        a,b and c parameters, needed for the energy.

    Returns
    -------
    float
        Energy
    """
    a,b,c = pars
    k_x,k_y = k_
    return -a*(1-b*(k_x**2+k_y**2))

def V_g(g,pars_moire):          #g is a integer from 0 to 5
    """Compute moire interlayer potential. 
    Not considering G and K different in Moire type for different orbitals.

    """
    N,V,phase,A_M = pars_moire
    return V*np.exp(1j*(-1)**g*phase)

def lu_table(N,G_M):
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

def big_H(K_,lu,pars_H,pars_moire,pars_grid,pars_interlayer,G_M):
    """Computes the large Hamiltonian containing all the moire replicas.

    """
    N,V,phase,A_M = pars_moire
    n_cells = int(1+3*N*(N+1))
    H_up = np.zeros((n_cells,n_cells),dtype=complex)
    H_down = np.zeros((n_cells,n_cells),dtype=complex)
    H_interlayer = np.zeros((n_cells,n_cells),dtype=complex)
    #
    for n in range(n_cells):      #circles go from 0 (central BZ) to N included
        KKK = K_ + G_M[0]*lu[n][0] + G_M[1]*lu[n][1]
        H_up[n:n+1,n:n+1] = energy_mono(KKK,pars_H[0])
        H_down[n:n+1,n:n+1] = energy_mono(KKK,pars_H[1])
        H_interlayer[n:n+1,n:n+1] = energy_inter(KKK,pars_interlayer)
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
                H_up[s:s+1,nn:nn+1] = V_g(g,pars_moire)
                H_down[s:s+1,nn:nn+1] = V_g(g,pars_moire)
    #All together
    final_H = np.zeros((2*n_cells,2*n_cells),dtype=complex)
    final_H[:n_cells,:n_cells] = H_up
    final_H[n_cells:,n_cells:] = H_down
    final_H[n_cells:,:n_cells] = H_interlayer
    final_H[:n_cells,n_cells:] = np.conjugate(H_interlayer.T)
    return final_H

def get_Moire(a_M):     
    """Compute Moire reciprocal lattice vectors.

    """
    G_M = [0,4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
    G_M[0] = np.tensordot(R_z(-np.pi/3),G_M[1],1)
    return G_M

def R_z(t):
    """Computes the matrix implementing z-rotatios.

    """
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R

def get_grid(pars_grid):
    """Compute the grid in momentum.

    """
    range_K, k_pts = pars_grid
    line = np.linspace(-range_K,range_K,k_pts)
    KX,KY = np.meshgrid(line,line)
    return (KX,KY)

def spread_lor(K,sp_k,k_list):
    """Computes the Lorentzian spread.

    """
    return 1/((k_list[:,None]-K[0])**2+(k_list[None,:]-K[1])**2+sp_k**2)**(3/2)

def spread_gauss(K,sp_k,k_list):
    """Computes the Gaussian spread.

    """
    return np.exp(-((K[0]-k_list[:,None])/sp_k)**2)*np.exp(-((K[1]-k_list[None,:])/sp_k)**2)

spread_fun_dic = {'Lorentz':spread_lor, 'Gauss':spread_gauss}

def normalize_cut(en_cut,pars_grid):
    """Normalize the energy cut in grayscale and put it in imshow format.

    """
    range_K,k_pts = pars_grid
    max_lor = np.max(np.ravel(en_cut))
    min_lor = np.min(np.ravel(np.nonzero(en_cut)))
    whitest = 255
    blackest = 0     
    norm_lor = np.zeros(en_cut.shape)
    for i in range(k_pts):
        for j in range(k_pts):
            norm_lor[i,j] = int((whitest-blackest)*(1-en_cut[i,j]/(max_lor-min_lor))+blackest)
    return np.flip(norm_lor.T,axis=0)   #invert e-axis

def compute_image_CEM(en_cut,en,max_E,pars_moire,pars_grid,pars_interlayer,pars_spread,cluster=False):
    """Compute CEM final image.

    """
    import matplotlib.pyplot as plt
    s_ = 20
    fig = plt.figure(figsize=(20,20))
    plt.imshow(en_cut,cmap='gray')
    plt.xlabel(r'$K_x$',size=s_)
    plt.ylabel(r'$K_y$',size=s_)
    title = "Energy from VBM: "+"{:.1f}".format(1000*abs(en-max_E))+" meV"
    plt.title(title,size=s_)
    plt.xticks([0,pars_grid[1]//2,pars_grid[1]],["{:.2f}".format(-pars_grid[0]/2),'0.00',"{:.2f}".format(pars_grid[0]/2)])
    plt.yticks([0,pars_grid[1]//2,pars_grid[1]],["{:.2f}".format(-pars_grid[0]/2),'0.00',"{:.2f}".format(pars_grid[0]/2)])
    fig = plt.gcf()
    fig_filename = final_fig_filename(en,pars_moire,pars_grid,pars_interlayer,pars_spread,cluster)
    fig.savefig(fig_filename)
    if not cluster:
        plt.show()
    plt.close()


#Filenames
def Hopt_filename(mat,n,v='2',cluster=False):
    """Computes the Hopt filename. 

    Parameters
    ----------
    mat: string
        material WS2 or WSe2.
    n: char
        The band to consider (2 for SO coupling).
    v: char (Default '2')
        version.
    cluster: bool
        machine.

    Returns
    -------
    string
        filename.
    """
    return home_dirname(cluster) + 'inputs/popt_'+mat+'_'+n+'_v'+v+'.npy'

def energies_filename(pars_moire,pars_grid,pars_interlayer,cluster=False):
    """Computes the energies filename. 

    Parameters
    ----------

    Returns
    -------
    string
        filename.
    """
    N,V,phase,A_M = pars_moire
    range_K, k_pts = pars_grid
    a,b,c = pars_interlayer
    return home_dirname(cluster) + 'results/energies_'+str(N)+'_'+"{:.3f}".format(V)+'_'+"{:.3f}".format(phase)+'_'+"{:.3f}".format(A_M)+'_'+"{:.3f}".format(range_K)+'_'+str(k_pts)+'_'+"{:.3f}".format(a)+'_'+"{:.3f}".format(b)+'_'+"{:.3f}".format(c)+'.npy'

def weights_filename(pars_moire,pars_grid,pars_interlayer,cluster=False):
    """Computes the weights filename. 

    Parameters
    ----------

    Returns
    -------
    string
        filename.
    """
    N,V,phase,A_M = pars_moire
    range_K, k_pts = pars_grid
    a,b,c = pars_interlayer
    return home_dirname(cluster) + 'results/weights_'+str(N)+'_'+"{:.3f}".format(V)+'_'+"{:.3f}".format(phase)+'_'+"{:.3f}".format(A_M)+'_'+"{:.3f}".format(range_K)+'_'+str(k_pts)+'_'+"{:.3f}".format(a)+'_'+"{:.3f}".format(b)+'_'+"{:.3f}".format(c)+'.npy'

def energy_cut_filename(en,pars_moire,pars_grid,pars_interlayer,pars_spread,cluster=False):
    """Computes the energy cut filename. 

    Parameters
    ----------

    Returns
    -------
    string
        filename.
    """
    N,V,phase,A_M = pars_moire
    range_K, k_pts = pars_grid
    a,b,c = pars_interlayer
    spread_k,spread_E,type_spread = pars_spread
    return home_dirname(cluster) + 'results/en_cut_'+"{:.4f}".format(en)+'_'+str(N)+'_'+"{:.3f}".format(V)+'_'+"{:.3f}".format(phase)+'_'+"{:.3f}".format(A_M)+'_'+"{:.3f}".format(range_K)+'_'+str(k_pts)+'_'+"{:.3f}".format(a)+'_'+"{:.3f}".format(b)+'_'+"{:.3f}".format(c)+'_'+"{:.3f}".format(spread_k)+'_'+"{:.3f}".format(spread_E)+'_'+type_spread+'.npy'

def final_fig_filename(en,pars_moire,pars_grid,pars_interlayer,pars_spread,cluster=False):
    """Computes the final figure filename. 

    Parameters
    ----------

    Returns
    -------
    string
        filename.
    """
    N,V,phase,A_M = pars_moire
    range_K, k_pts = pars_grid
    a,b,c = pars_interlayer
    spread_k,spread_E,type_spread = pars_spread
    return home_dirname(cluster) + 'results/figure_'+"{:.4f}".format(en)+'_'+str(N)+'_'+"{:.3f}".format(V)+'_'+"{:.3f}".format(phase)+'_'+"{:.3f}".format(A_M)+'_'+"{:.3f}".format(range_K)+'_'+str(k_pts)+'_'+"{:.3f}".format(a)+'_'+"{:.3f}".format(b)+'_'+"{:.3f}".format(c)+'_'+"{:.3f}".format(spread_k)+'_'+"{:.3f}".format(spread_E)+'_'+type_spread+'.png'

def home_dirname(cluster=False):
    """Computes the home dirname.

    Parameters
    ----------
    cluster: bool
        machine.

    Returns
    -------
    string
        dirname.
    """
    return "/home/users/r/rossid/4_CEM_K/" if cluster else "/home/dario/Desktop/git/MoireBands/4_CEM_K/"

def tqdm(x):
    return x

