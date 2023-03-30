import functions as fs
import parameters as PARS
import sys
import numpy as np
import getopt
import scipy.linalg as la
from time import time as tt

dirname = "/home/users/r/rossid/moire_Data/"                    #WRONG
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:J:",["LL=","UL=","pts_ps="])
    J = 0               #index of columns
    N = 0               #Number of circles of mini-BZ around the central one
    upper_layer = 'WSe2'
    lower_layer = 'WS2'
    pts_ps = 20         #points per step
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt in ['-J']:
        J = int(arg)
    if opt == '--LL':
        lower_layer = arg
    if opt == '--UL':
        upper_layer = arg
    if opt == '--pts_ps':
        pts_ps = int(arg)

if J > pts_ps*2-1:
    print("J out of bounds")
    exit()
params_H =  [PARS.dic_params_H[upper_layer], PARS.dic_params_H[lower_layer]]
params_SO = [PARS.dic_params_SO[upper_layer[0]], PARS.dic_params_SO[upper_layer[1:]],PARS.dic_params_SO[lower_layer[0]], PARS.dic_params_SO[lower_layer[1:]] ]
params_V =  [PARS.dic_params_V[upper_layer+'/'+lower_layer], PARS.dic_params_V[lower_layer+'/'+upper_layer]]
a_M =       PARS.dic_a_Moire[upper_layer+'/'+lower_layer]
a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]

#define k-points to compute
G_mono = [4*np.pi/np.sqrt(3)/a_mono[0]*np.array([0,1])]    #reciprocal lattice vector of upper monolayer
for i in range(1,6):
    G_mono.append(np.tensordot(fs.R_z(np.pi/3*i),G_mono[0],1))    #construct the others by rotating
K = np.array([G[-1][0]/3*2,0])                          #K-point
Gamma = np.array([0,0])                                 #Gamma-point
M =     G[-1]/2                                         #M-point
Kp =    np.tensordot(fs.R_z(np.pi/3),K,1)               #K'-point
#
K_initial = np.array([-K[0],-Kp[1]]) + J/(pts_ps*2)*np.array([2*K[0],0])
K_final = K_initial + np.array([0,2*Kp[1]])
path = np.linspace(K_initial,K_final,2*pts_ps)
#Moiré reciprocal lattice vectors. I start from the first one and obtain the others by doing pi/3 rotations
G_M = [4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
for i in range(1,6):
    G_M.append(np.tensordot(R_z(np.pi/3*i),G_M[0],1))

######################
###################### Construct Hamiltonians with Moirè potential
######################
n_cells = int(1+3*N*(N+1))*2        #Dimension of H divided by 3 -> take only valence bands
sbv = [-10,10]                      #Select_by_value for the diagonalization in order to take only bands in valence. 
                                    #Needs to be fitted
res = np.zeros((2,len(path),n_cells))           #Energies: 2 -> layers, len(path) -> k-points, n_cells -> dimension of Hamiltonian
weight = np.zeros((2,len(path),n_cells))        #ARPES weights
for i in range(len(path)):
    K = path[i]                                 #Considered K-point
    H_UL = fs.total_H(K,N,params_H[0],params_V[0:1],G_M,a_mono[0])     #Compute UL Hamiltonian for given K
    H_LL = fs.total_H(K,N,params_H[1],params_V[2:3],G_M,a_mono[1])     #Compute LL Hamiltonian for given K
    res[0,i,:],evecs_UL = la.eigh(H_UL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
    res[1,i,:],evecs_LL = la.eigh(H_LL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
    evecs = [evecs_UL,evecs_LL]
    for l in range(2):
        for e in range(n_cells):
            for d in range(6):
                weight[l,i,e] += np.abs(evecs[l][d,e])**2
#
res_mono_UL = np.zeros((len(path),6))
res_mono_LL = np.zeros((len(path),6))
params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
for i in range(len(path)):
    K = path[i]
    H_UL = fs.total_H(K,0,params_H[0],params_V[0:1],G_M,a_mono[0])     #Compute UL Hamiltonian for given K
    H_LL = fs.total_H(K,0,params_H[1],params_V[2:3],G_M,a_mono[1])     #Compute LL Hamiltonian for given K
    res_mono_UL[i,:],evecs_mono = np.linalg.eigh(H_UL)
    res_mono_LL[i,:],evecs_mono = np.linalg.eigh(H_LL)
############
data_name = dirname+"energies_"+lower_layer+"-"+upper_layer+"_"+str(J)+".npy"
weights_name = dirname+"arpes_"+lower_layer+"-"+upper_layer+"_"+str(J)+".npy"    
mono_LL_name = dirname+"mono_"+lower_layer+'_'+str(J)+".npy"
mono_UL_name = dirname+"mono_"+upper_layer+'_'+str(J)+".npy"
#
np.save(data_name,res)
np.save(weights_name,weight)
np.save(mono_UL_name,res_mono_UL)
np.save(mono_LL_name,res_mono_LL)
##################


























