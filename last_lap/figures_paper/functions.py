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

def get_K(cut,n_pts):
    res = np.zeros((n_pts,2))
    a_mono = 1
    if cut == 'KGK':
        Ki = -np.array([4*np.pi/3,0])
        Kf = np.array([4*np.pi/3,0])
    elif cut == 'MGM':
        Ki = -np.array([2*np.pi,2*np.pi/np.sqrt(3)])/2
        Kf = np.array([2*np.pi,2*np.pi/np.sqrt(3)])/2
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
def gap(energies,weights,i_k,list_momenta):
    """
    Energy gap between two neighboring bands. 
    level specifies which bands from the top are considered.
    G_pt specifies in which momentum the gap is considered.
    """
    levs = np.argsort(weights[i_k,:])
    upper_level = levs[-1]
    lower_level = levs[::-1][np.argmax(levs[::-1]<levs[-1])]
    return (energies[:,upper_level]-energies[:,lower_level])[i_k], upper_level, lower_level
#band distance
def horizontal_displacement(e_,energies,weights,list_momenta,mass):
    """
    Compute indeces of main and first two side bands, using the weights to discriminate.
    """
    #distance in energy between 2 k ppoints on same band
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
    i_mb = filtered_indices[0]
    i_sb1 = filtered_indices[2]             #BECAUSE THERE IS A SIDE BAND VERY CLOSE TO THE MAIN ONE!!!!
    i_sb2 = filtered_indices[3]
    return l,(i_mb,i_sb1,i_sb2)

def vertical_displacement(i_k,weights):
    indices = np.argsort(weights[i_k,:])
    i_mb = indices[-1]
    i_sb2 = indices[-3]
    i_sb1 = indices[-4] #if indices[-3] < indices[-2] else indices[-4]
    return (i_mb,i_sb1,i_sb2)

def plot_single_parameter_set(e_,energies,weights,pars,list_momenta,title):
    """
    Given energy e_ and a set of eigenvalues and weights computes the image.
    """
    N,V,phi,mass,mrl,cut = pars
    #Figure
    fig,ax = plt.subplots()
    fig.set_size_inches(18,12)
    LW = 0.1
    abs_k = np.array([np.linalg.norm(list_momenta[i])*np.sign(list_momenta[i,0]) for i in range(list_momenta.shape[0])])
    for t in range(energies.shape[1]):
        ax.plot(abs_k,energies[:,t],'k-',lw=LW)
        ax.scatter(abs_k,energies[:,t],s=weights[:,t]*20,c='b',lw=0)
    #Gap arrows
    gap_k = -mrl/np.sqrt(3) if cut == 'KGK' else -mrl/2
    ind_gapk = ind_k(gap_k,list_momenta)
    E_gap, up_l, low_l = gap(energies,weights,ind_gapk,list_momenta)
    ax.plot([abs_k[ind_gapk],abs_k[ind_gapk]],
            [energies[ind_gapk,low_l],energies[ind_gapk,up_l]],
            color='r',label='gap 1')
    if 0:
        gap_k *= 3/2
        ind_gapk = ind_k(gap_k,list_momenta)
        E_gap, up_l, low_l = gap(energies,weights,ind_gapk,list_momenta)
        ax.plot([abs_k[ind_gapk],abs_k[ind_gapk]],
                [energies[ind_gapk,low_l],energies[ind_gapk,up_l]],
                color='maroon',label='gap 2')
    #Horizontal distance arrow
    l, inds = horizontal_displacement(e_,energies,weights,list_momenta,mass)
    i_mb,i_sb1,i_sb2 = inds
#    ax.scatter(abs_k[l[i_mb,0]],energies[l[i_mb,0],l[i_mb,1]],c='k',s=150)
    ax.scatter(abs_k[l[i_sb1,0]],energies[l[i_sb1,0],l[i_sb1,1]],c='lime',s=30)
    ax.scatter(abs_k[l[i_sb2,0]],energies[l[i_sb2,0],l[i_sb2,1]],c='g',s=30)

    ax.arrow(abs_k[l[i_mb,0]],energies[l[i_mb,0],l[i_mb,1]],
            abs_k[l[i_sb1,0]]-abs_k[l[i_mb,0]],0,
            color='lime',label='h displacement 1',head_length=0,width=0)
    ax.arrow(abs_k[l[i_mb,0]],energies[l[i_mb,0],l[i_mb,1]],
            abs_k[l[i_sb2,0]]-abs_k[l[i_mb,0]],0,
            color='g',label='h displacement 2',head_length=0,width=0)
    #Vertical distance arrow
    i_k = l[i_mb,0]
    i_mb,i_sb1,i_sb2 = vertical_displacement(i_k,weights)
    ax.scatter(abs_k[i_k],energies[i_k,i_mb],c='m',s=50,zorder=10)
    ax.scatter(abs_k[i_k],energies[i_k,i_sb1],c='aqua',s=30)
    ax.scatter(abs_k[i_k],energies[i_k,i_sb2],c='dodgerblue',s=30)

    ax.arrow(abs_k[i_k],energies[i_k,i_mb],     #x,y,dx,dy
            0,energies[i_k,i_sb1]-energies[i_k,i_mb],
            color='aqua',label='v displacement 1',head_length=0,width=0)
    ax.arrow(abs_k[i_k],energies[i_k,i_mb],
            0,energies[i_k,i_sb2]-energies[i_k,i_mb],
            color='dodgerblue',label='v displacement 2',head_length=0,width=0)

    #Plot features
    ax.legend()
    ax.set_title(title)
    #Limits
    rg = np.max(energies[:,-1])-np.min(energies[:,N])
    ax.set_ylim(e_*3,np.max(energies[:,-1])*2)
    plt.show()




