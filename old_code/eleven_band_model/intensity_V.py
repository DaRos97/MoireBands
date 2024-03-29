import functions as fs
import parameters as PARS
import sys
import numpy as np
import getopt
import scipy.linalg as la
from time import time as tt

####not in cluster
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

dirname = "../Data/11_bands/"                    #WRONG
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["LL=","UL=","ap"])
    N = 1               #Number of circles of mini-BZ around the central one
    upper_layer = 'WSe2'
    lower_layer = 'WS2'
    ap = 'amplitude'       #amplitude
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt == '--LL':
        lower_layer = arg
    if opt == '--UL':
        upper_layer = arg
    if opt == '--ap':
        ap = 'phase'  #phase

#
hopping = [PARS.find_t(upper_layer),PARS.find_t(lower_layer)]
epsilon = [PARS.find_e(upper_layer),PARS.find_e(lower_layer)]
HSO = [PARS.find_HSO(upper_layer),PARS.find_HSO(lower_layer)]
#
a_M =       PARS.dic_a_Moire[upper_layer+'/'+lower_layer]
a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]

#define k-point
G = [4*np.pi/np.sqrt(3)/a_mono[0]*np.array([0,1])]      
for i in range(1,6):
    G.append(np.tensordot(fs.R_z(np.pi/3*i),G[0],1))
#
K_pt = np.array([G[-1][0]/3*2,0])                      #K-point
#Moiré reciprocal lattice vectors. I start from the first one and obtain the others by doing pi/3 rotations
G_M = [4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
for i in range(1,6):
    G_M.append(np.tensordot(fs.R_z(np.pi/3*i),G_M[0],1))

#Moirè potential points to compute
params_V =  PARS.dic_params_V[upper_layer+'/'+lower_layer]
n_pts = 100
end_V = 2*params_V[2] if ap=='amplitude' else 2*np.pi
V_path = np.linspace(0,end_V,n_pts)
moire_vector = 0        #index of Moire reciprocal lattice vector

######################
###################### Construct Hamiltonians with Moirè potential
######################
n_cells = int(1+3*N*(N+1))*14        #Dimension of H divided by 3 -> take only valence bands
sbv = [-10,1]                      #Select_by_value for the diagonalization in order to take only bands in valence. 
res_name = dirname + "int_V_arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+"_"+str(moire_vector)+ap+".npy"
try:    #name: LL/UL, N, Path, k-points per segment
    res = np.load(res_name)
    print("\nIntensities already computed")
except:
    ti = tt()
    weight1 = np.zeros((2,n_pts,n_cells))        #ARPES weights
    weight2 = np.zeros((2,n_pts,n_cells))        #ARPES weights
    res = np.zeros((2,n_pts,n_cells))        #ARPES weights
    for i in tqdm.tqdm(range(n_pts)):
        ind_ap = 2 if ap=='amplitude' else 3        #index of varying parameter in params_V
        params_V[ind_ap] = V_path[i]
        #First point
        K1 = K_pt                                #Considered K-point
        H_UL = fs.total_H(K1,N,hopping[0],epsilon[0],HSO[0],params_V,G_M,a_mono[0])     #Compute UL Hamiltonian for given K
        H_LL = fs.total_H(K1,N,hopping[1],epsilon[1],HSO[1],params_V,G_M,a_mono[1])     #Compute LL Hamiltonian for given K
        res0,evecs_UL = la.eigh(H_UL,subset_by_value=sbv)           #evec_UL has (1+3*N*(N+1))*22 lines and (1+3*N*(N+1))*14 columns -> eqch column is an eigenvector of H_UL
        res1,evecs_LL = la.eigh(H_LL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
        evecs = [evecs_UL,evecs_LL]
        for l in range(2):
            for e in range(n_cells):
                for d in range(22):
                    weight1[l,i,e] += np.abs(evecs[l][d,e])**2
        #Second point
        K2 = K_pt + G_M[moire_vector]                                #Considered K-point
        H_UL = fs.total_H(K2,N,hopping[0],epsilon[0],HSO[0],params_V,G_M,a_mono[0])     #Compute UL Hamiltonian for given K
        H_LL = fs.total_H(K2,N,hopping[1],epsilon[1],HSO[1],params_V,G_M,a_mono[1])     #Compute LL Hamiltonian for given K
        res0,evecs_UL = la.eigh(H_UL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
        res1,evecs_LL = la.eigh(H_LL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
        evecs = [evecs_UL,evecs_LL]
        for l in range(2):
            for e in range(n_cells):
                for d in range(22):
                    weight2[l,i,e] += np.abs(evecs[l][d,e])**2
        #Compute weight ratio
        for j in range(n_cells):
            if weight1[0,i,j] != 0:
                res[0,i,j] = weight2[0,i,j]/weight1[0,i,j]
            if weight1[1,i,j] != 0:
                res[1,i,j] = weight2[1,i,j]/weight1[1,i,j]
    np.save(res_name,res)
    print("Time taken: ",tt()-ti)

plt.figure()
for i in [0,-1]:
    plt.plot(V_path,res[0,:,i],'*-',label=str(i))
plt.legend()
if ap == 'amplitude':
    plt.ylim([0,1])
ggg = 15
label_x = r'$|V_k|$' if ap=='amplitude' else r'$\phi_k$'
plt.xlabel(label_x,fontsize=ggg)
plt.ylabel(r'$\frac{I(K+G_0)}{I(K)}$',rotation=0,fontsize=ggg)
plt.show()






