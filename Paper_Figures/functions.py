import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

a_1 = np.array([1,0])
a_2 = np.array([-1/2,np.sqrt(3)/2])

def big_H(K_,lu,pars,G_M):
    """Computes the large Hamiltonian containing all the moire replicas.

    """
    N,V,phi,mass = pars
    n_cells = int(1+3*N*(N+1))          #Number of mBZ copies
    H_up = np.zeros((n_cells,n_cells),dtype=complex)
    #Diagonal parts
    for n in range(n_cells):
        Kn = K_ + G_M[0]*lu[n][0] + G_M[1]*lu[n][1]
        H_up[n,n] = -np.linalg.norm(Kn)**2/2/mass
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
    return H_up

def get_V(V,phi,ind):
    return V*np.exp(1j*(-1)**(ind%2)*phi)

def get_K(cut,n_pts,aM):
    res = np.zeros((n_pts,2))
    if cut == 'KGK':
        Ki = -np.array([4*np.pi/3,0])/aM*5
    elif cut == 'MGM':
        Ki = -np.array([2*np.pi,2*np.pi/np.sqrt(3)])/2/aM*5
    Kf = -Ki
    for i in range(n_pts):
        res[i] = Ki+(Kf-Ki)*i/n_pts
    return res

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

def R_z(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

#Gap
def ind_k(k_pt,list_momenta):
    """
    Index of k_pt in momentum list.
    """
    initial_momentum = list_momenta[0]
    final_momentum = list_momenta[-1]
    momentum_points = len(list_momenta)
    return int(momentum_points*np.linalg.norm(k_pt-initial_momentum)/np.linalg.norm(final_momentum-initial_momentum))
def gap(energies,weights,i_k,cut):
    """
    Energy gap between two neighboring bands. 
    level specifies which bands from the top are considered.
    G_pt specifies in which momentum the gap is considered.
    """
    levs = np.argsort(weights[i_k,:])
    upper_level = levs[-1]
    lower_level = levs[-2]
    g1 = abs((energies[:,-1]-energies[:,-2])[i_k])
    g2 = abs((energies[:,-2]-energies[:,-3])[i_k])
    gg = max(g1,g2)
    n_b = weights.shape[1]-1
    upper_level,lower_level = (n_b,n_b-1) if g1>g2 else (n_b-1,n_b-2)
    if cut == 'KGK':
        return gg, upper_level, lower_level
    else:
        return g1, n_b, n_b-1
#band distance
def horizontal_displacement(e_,energies,weights,list_momenta,mass):
    """
    Compute indeces of main and first two side bands, using the weights to discriminate.
    """
    #distance in energy between 2 k points on same band
    delta_e = 1/2/mass*np.sqrt(2*mass*abs(e_))*np.linalg.norm(list_momenta[1]-list_momenta[0])
    momentum_points = list_momenta.shape[0]
    l = np.argwhere(abs(e_-energies[:momentum_points//2,:])<delta_e)
    indices = np.argsort([weights[l[i,0],l[i,1]] for i in range(l.shape[0])])
    #
    filtered_indices = [indices[-1],]
    for i in range(indices.shape[0]):
        temp = indices[-1-i]
        if abs(l[temp,0]-l[filtered_indices[-1],0])>2:
                filtered_indices.append(temp)
    #Selection of indices
    i_mb = filtered_indices[0]
    if filtered_indices[1] > i_mb:
        i_sb2 = filtered_indices[1]
        a = 1
    else:
        i_sb1 = filtered_indices[1]
        a = 0
    for i in range(2,len(filtered_indices)):
        if filtered_indices[i] < i_mb and a:
            i_sb1 = filtered_indices[i]
            break
        elif filtered_indices[i] > i_mb and not a:
            i_sb2 = filtered_indices[i]
            break
    return l,(i_mb,i_sb1,i_sb2) #sb1->external

def vertical_displacement(i_k,weights):
    indices = np.argsort(weights[i_k,:])
    #Selection of indices
    i_mb = indices[-1]
    if indices[-2] < i_mb:
        i_sb1 = indices[-2]
        a = 1
    else:
        i_sb2 = indices[-2]
        a = 0
    for i in range(-3,-len(indices),-1):
        if indices[i] > i_mb and a:
            i_sb2 = indices[i]
        elif indices[i] < i_mb and not a:
            i_sb1 = indices[i]
            break
    return (i_mb,i_sb1,i_sb2) #sb1->lower

def plot_single_parameter_set(e_,energies,weights,pars,list_momenta,title):
    """
    Given energy e_ and a set of eigenvalues and weights computes the image.
    """
    N,V,phi,mass,mrl,cut,G_M = pars
    #Figure
    fig,ax = plt.subplots()
    fig.set_size_inches(18,12)
    LW = 0.05
    abs_k = np.array([np.linalg.norm(list_momenta[i])*np.sign(list_momenta[i,0]) for i in range(list_momenta.shape[0])])
    for t in range(energies.shape[1]):
        ax.plot(abs_k,energies[:,t],'k-',lw=LW)
        ax.scatter(abs_k,energies[:,t],s=weights[:,t]*100,c='b',lw=0)
    if not V==0:
        #Gap
        k_pt = np.array([-mrl/np.sqrt(3),0]) if cut == 'KGK' else -G_M[0]/2
        ind_gapk = ind_k(k_pt,list_momenta)
        gap_k = np.linalg.norm(k_pt)
        E_gap, up_l, low_l = gap(energies,weights,ind_gapk,cut)
        ax.plot([abs_k[ind_gapk],abs_k[ind_gapk]],
                [energies[ind_gapk,low_l],energies[ind_gapk,up_l]],
                color='r',label='gap 1')
        if 0:   #Second gap
            gap_k *= 3/2
            ind_gapk = ind_k(gap_k,list_momenta)
            E_gap, up_l, low_l = gap(energies,weights,ind_gapk,cut)
            ax.plot([abs_k[ind_gapk],abs_k[ind_gapk]],
                    [energies[ind_gapk,low_l],energies[ind_gapk,up_l]],
                    color='maroon',label='gap 2')
        #Horizontal bands
        l, inds = horizontal_displacement(e_,energies,weights,list_momenta,mass)
        i_mb,i_sb1,i_sb2 = inds
        ax.scatter(abs_k[l[i_sb1,0]],energies[l[i_sb1,0],l[i_sb1,1]],c='lime',s=30)
        ax.scatter(abs_k[l[i_sb2,0]],energies[l[i_sb2,0],l[i_sb2,1]],c='g',s=30)
        
        ax.hlines(energies[l[i_mb,0],l[i_mb,1]],    #y
                abs_k[l[i_mb,0]],abs_k[l[i_sb1,0]], #xmin,xmax
                color='lime',label='external band')
        ax.hlines(energies[l[i_mb,0],l[i_mb,1]],    #y
                abs_k[l[i_mb,0]],abs_k[l[i_sb2,0]], #xmin,xmax
                color='g',label='internal band')
        #Vertical bands
        i_k = l[i_mb,0]
        i_mb,i_sb1,i_sb2 = vertical_displacement(i_k,weights)
        ax.scatter(abs_k[i_k],energies[i_k,i_mb],c='m',s=50,zorder=10)
        ax.scatter(abs_k[i_k],energies[i_k,i_sb1],c='aqua',s=30)
        ax.scatter(abs_k[i_k],energies[i_k,i_sb2],c='dodgerblue',s=30)

        ax.plot([abs_k[i_k],abs_k[i_k]],
                [energies[i_k,i_mb],energies[i_k,i_sb1]],
                color='aqua',label='lower band')
        ax.plot([abs_k[i_k],abs_k[i_k]],
                [energies[i_k,i_mb],energies[i_k,i_sb2]],
                color='dodgerblue',label='higher band')
        #Plot features
        ax.legend(fontsize=20,loc='upper right')
    ax.set_title(title,size=30)
    #Limits
    rg = np.max(energies[:,-1])-np.min(energies[:,N])
    ax.set_ylim(e_*2,abs(e_/2))
    ax.set_xlim(-2.8*mrl,2.8*mrl)
    ax.set_xticks([-mrl/2*np.sqrt(3),0,mrl/2*np.sqrt(3)],[r'$-\tilde{M}$',r'$\Gamma$',r'$\tilde{M}$'],size=30)
    ax.plot([0,0],[-10,10],color='k',lw=0.5,zorder=0)
    ax.plot([-mrl/2*np.sqrt(3),-mrl/2*np.sqrt(3)],[-10,10],color='k',lw=0.5,zorder=0)
    ax.plot([mrl/2*np.sqrt(3),mrl/2*np.sqrt(3)],[-10,10],color='k',lw=0.5,zorder=0)
    ax.set_yticks([])
    ax.set_ylabel('Energy',size=30)
#    ax.set_xticks([])
#    ax.set_yticks([])
    plt.show()




