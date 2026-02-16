import numpy as np
import itertools

def get_pars(ind):
    lMonolayer_type = ['fit',]
    lInterlayer_symm = ['C6',]
    pars_Vgs = [0.02,]#np.linspace(0.001,0.03,11)#[0.02,]        #Moire potential at Gamma
    pars_Vks = [0.0077,]#[0.0077,]             #Moire potential at K
    phi_G = [0*np.pi,]#np.linspace(0,np.pi,7)#[0,np.pi,]                #Phase at Gamma
    phi_K = [-106*np.pi/180,]       #Phase at K
    lSample = ['S3',]  #this and theta are related! Sample needed also for interlayer parameters' choice
    lTheta = [0,]#2.8,] if lSample[0]=='S11' else [1.8,]    #twist angle
    lN = [1,]                       #number of BZ circles
    lKpts = [6,]           #number of k points in each direction of the miniBZ -> computing k_pts**2 momenta
    lRpts = [300,]          #number of r points to compute
    #
    ll = [lMonolayer_type,lInterlayer_symm,pars_Vgs,pars_Vks,phi_G,phi_K,lTheta,lSample,lN,lKpts,lRpts]
    return list(itertools.product(*ll))[ind]

def compute_kList(kPts,G_M):
    """
    Here we have to define the grid of momentum points to sum over.
    Here now is a grid of the mini-BZ.
    """
    G1,G2 = G_M
    kList = np.zeros((kPts,kPts,2))
    for ix in range(kPts):
        for iy in range(kPts):
            kList[ix,iy] = G1*ix/kPts + G2*iy/kPts
    kFlat = kList.reshape(-1,2)
    return kFlat

def compute_kBands(l1,l2,G_M):
    """
    Here we have to define the cut of momentum points for plotting the bands.
    We go k'+ -> g -> k- -> k+ -> k'+.
    k'+ is a 120°-rotated k+.
    """
    G1,G2 = G_M
    k1 = (G1+G2)/3
    k_pp = R_z(np.pi/3*5) @ k1
    k_m = R_z(np.pi/3*2) @ k1
    k_p = R_z(np.pi/3*3) @ k1
    g = np.zeros(2)
    list_ = np.zeros((3*l1+l2,2))
    for i in range(l1):
        list_[i] = k_pp + i/l1 * (g-k_pp)
    for i in range(l1):
        list_[i+l1] = g + i/l1 * (k_m-g)
    for i in range(l1):
        list_[i+2*l1] = k_m + i/l1 * (k_p-k_m)
    for i in range(l2):
        list_[i+3*l1] = k_p + i/l2 * (k_pp-k_p)
    return list_

def compute_rList(rPts,a_M):
    """
    Here we have to define the real space points to compute.
    """
    a1,a2 = a_M
    rList = np.zeros((rPts,2))
    for i in range(rPts):
        rList[i] = (a1+a2)*i/rPts
    return rList

def get_moire(G_M):
    """
    Here we compute the moirè real-space vectors from the reciprocal ones.
    """
    a_M = 2*np.pi*np.linalg.inv(np.array(G_M))
    a1 = a_M[:,0]
    a2 = R_z(np.pi/3) @ a1
    return [a1,a2]

def get_fn(*args):
    """
    Get filename for set of parameters.
    """
    fn = ''
    for i,a in enumerate(args):
        t = type(a)
        if t in [str,]:
            fn += a
        elif t in [int, np.int64]:
            fn += str(a)
        elif t in [float, np.float32, np.float64]:
            fn += "{:.7f}".format(a)
        if not i==len(args)-1:
            fn +='_'
    return fn


def R_z(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

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

def H_pan(k,*args):
    """
    Matrix Hamiltonian.
    k is 2-D.
    """
    lu,nShells,nCells,en_coeff,bs,Vm,w = args
    k_p = (-bs[1]-bs[2])/3
    k_m = (-2*bs[1]+bs[2])/3
    mat = np.zeros((2*nCells,2*nCells),dtype=complex)
    if 0:#Moire
        mat[0,1] = mat[0,3] = mat[0,5] = Vm
        mat[0,2] = mat[0,4] = mat[0,6] = Vm.conj()
        mat[1,2] = mat[1,6] = mat[3,4] = mat[5,6] = Vm
        mat[2,3] = mat[4,5] = Vm.conj()
        mat += mat.T.conj()
    #Moire
    m = [[-1,1],[-1,0],[0,-1],[1,-1],[1,0],[0,1]]
    for n in range(0,nShells+1):      #Circles
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       #Indices inside the circle
            ind_s = lu[s]
            for i in m:
                ind_nn = (ind_s[0]+i[0],ind_s[1]+i[1])  #nn-> nearest neighbour
                try:
                    nn = lu.index(ind_nn)
                except:
                    continue
                g = m.index(i)
                Vup = Vm if g%2 else Vm.conj()
                mat[s:s+1,nn:nn+1] = Vup #H_moires[g%2]    #H_moire(g,pars_moire[1])
                mat[nCells+s:nCells+s+1,nCells+nn:nCells+nn+1] = Vup.conj() #H_moires[g%2]    #H_moire(g,pars_moire[1])
                #Interlayer
#                mat[s:s+1,nCells+nn:nCells+nn+1] = w #H_moires[g%2]    #H_moire(g,pars_moire[1])
    Kns = np.zeros((nCells,2))
    for i in range(nCells):
        Kns[i] = k + bs[1]*lu[i][0] + bs[2]*lu[i][1]
    for i in range(nCells):
        #Dispersion
        Kn = Kns[i]
        mat[i,i]                = -en_coeff*np.linalg.norm(Kn - k_p)**2
        mat[i+nCells,i+nCells]  = -en_coeff*np.linalg.norm(Kn - k_m)**2
        #Interlayer
        for nb in [0,2,3]:
            diff = np.linalg.norm(np.absolute(Kn - Kns + bs[nb]),axis=1)
#            print(i,nb)
#            print(diff)
            inds = np.where(diff<1e-10)[0]
#            print(inds)
#            input()
            for ind in inds:
                mat[i,ind+nCells] += w
#                mat[ind,i+nCells] += w
    mat[nCells:,:nCells] = mat[:nCells,nCells:].T
    return mat














