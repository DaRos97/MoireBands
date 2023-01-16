import functions as fs
import parameters as PARS
import sys
import numpy as np
import getopt
import scipy.linalg as la
from time import time as tt

dirname = "~/"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:J:",["plot","LL=","UL=","path=","pts_ps=","fc","method=","EnGrid=","mono","offset_energy="])
    J = 0               #index of columns
    N = 1
    lower_layer = 'WSe2'
    upper_layer = 'WS2'
    Path = 'KGC'               #Points of BZ-path
    pts_ps = 50         #points per step
    plot = False
    FC = False                  #False Color plot
    method = 'GGA'
    gridy = 0
    mono = False
    offset_energy = -0.41#in eV
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt in ['-J']:
        J = int(arg)
    if opt == '--plot':
        plot = True
    if opt == '--LL':
        lower_layer = arg
    if opt == '--UL':
        upper_layer = arg
    if opt == '--path':
        Path = arg
    if opt == '--pts_ps':
        pts_ps = int(arg)
    if opt == '--fc':
        FC = True
    if opt == '--method':
        method = arg
    if opt == '--EnGrid':
        gridy = int(arg)
    if opt == '--mono':
        mono = True
    if opt == '--offset_energy':
        offset_energy = float(arg)

if J > pts_ps*2-1:
    print("J out of bounds")
    exit()
if gridy == 0:
    gridy = pts_ps*(len(Path)-1)
params_H =  [PARS.dic_params_H[method][upper_layer], PARS.dic_params_H[method][lower_layer]]
params_V =  [PARS.dic_params_V[upper_layer+'/'+lower_layer], PARS.dic_params_V[lower_layer+'/'+upper_layer]]
a_M =       PARS.dic_a_M[upper_layer+'/'+lower_layer]

#define k-ts
a_monolayer = params_H[0][0] #lattice length of upper layer
G = [4*np.pi/np.sqrt(3)/a_monolayer*np.array([0,1])]      
for i in range(1,6):
    G.append(np.tensordot(fs.R_z(np.pi/3*i),G[0],1))

K = np.array([G[-1][0]/3*2,0])                      #K-point
Gamma = np.array([0,0])                                #Gamma
M =     G[-1]/2                          #M-point
Kp =    np.tensordot(fs.R_z(np.pi/3),K,1)     #K'-point

K_initial = np.array([-K[0],-Kp[1]]) + J/(pts_ps*2)*np.array([2*K[0],0])
K_final = K_initial + np.array([0,2*Kp[1]])
path = np.linspace(K_initial,K_final,2*pts_ps)
#
n_cells = int(1+3*N*(N+1))*2        #dimension of H divided by 3 -> take only valence bands     #Full Diag -> *3
sbv = [-2,0.5]                      #select_by_value for the diagonalization -> takes only bands in valence
res = np.zeros((2,len(path),n_cells))
weight = np.zeros((2,len(path),n_cells))
for i in range(len(path)):
    K = path[i]
    H_UL = fs.total_H(K,N,params_H[0],params_V[0],a_M)     #Compute Hamiltonian for given K
    H_LL = fs.total_H(K,N,params_H[1],params_V[1],a_M)     #Compute Hamiltonian for given K
    res[0,i,:],evecs_UL = la.eigh(H_UL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
    res[1,i,:],evecs_LL = la.eigh(H_LL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
    evecs = [evecs_UL,evecs_LL]
    for l in range(2):
        for e in range(n_cells):
            for d in range(6):
                weight[l,i,e] += np.abs(evecs[l][d,e])**2
res[1] -= offset_energy
#
res_mono_UL = np.zeros((len(path),6))
res_mono_LL = np.zeros((len(path),6))
params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
for i in range(len(path)):
    K = path[i]
    H_k_UL = fs.total_H(K,0,params_H[0],params_V,a_M)     #the only difference is in N which now is 0
    H_k_LL = fs.total_H(K,0,params_H[1],params_V,a_M)     #the only difference is in N which now is 0
    res_mono_UL[i,:],evecs_mono = np.linalg.eigh(H_k)
    res_mono_LL[i,:],evecs_mono = np.linalg.eigh(H_k)
############
data_name = dirname+"res_"+lower_layer+"-"+upper_layer+"_"+str(J)+".npy"
weights_name = dirname+"arpes_"+lower_layer+"-"+upper_layer+"_"+str(J)+".npy"    
mono_LL_name = dirname+"mono_"+lower_layer+'_'+str(J)+".npy"
mono_UL_name = dirname+"mono_"+upper_layer+'_'+str(J)+".npy"
#
np.save(data_name,res)
np.save(weights_name,weight)
np.save(mono_UL_name,res_mono_UL)
np.save(mono_LL_name,res_mono_LL)
##################


























