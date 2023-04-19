import numpy as np
import matplotlib.pyplot as plt
import parameters as ps
import scipy.linalg as la
from time import time as tt

def convert(input_filename):
    with open(input_filename, 'r') as f:
        lines = f.readlines()
    N = len(lines)
    res = np.ndarray((N,2))
    for i in range(N):
        temp = lines[i].split('\t')
        res[i,0] = float(temp[0])
        res[i,1] = float(temp[1])
    return res

def reduce_input(input_data,considered_pts):
    N = len(input_data[0][:,0])
    new_N = N//(N//considered_pts)
    res = []
    res.append(np.ndarray((new_N,2)))
    res.append(np.ndarray((new_N,2)))
    for i in range(new_N):
        ind = N//considered_pts*i
        res[0][i,:] = input_data[0][ind,:]
        res[1][i,:] = input_data[1][ind,:]
    return res

def find_vec_k(k_scalar,path):
    k_pts = np.ndarray((len(k_scalar),2))
    if path == 'KGC':
        for i in range(len(k_scalar)):
            if k_scalar[i] < 0:
                k_pts[i,0] = np.abs(k_scalar[i])
                k_pts[i,1] = 0
            else:
                k_pts[i,0] = np.abs(k_scalar[i])*np.cos(np.pi/3)
                k_pts[i,1] = np.abs(k_scalar[i])*np.sin(np.pi/3)
    else:
        print("Path not implemented in vectorization of k points")
        exit()
    return k_pts

def chi2(parameters,*args):
    ti = tt()
    input_energy, M, a_mono, N, k_pts,k2,plot = args
    energies_computed = energies(parameters,M,a_mono,k_pts)
    res = 0
    for band in range(2):
        for i in range(N):
            res += (energies_computed[band,i] - input_energy[band][i])**2
    if res < ps.list_res_bm[ps.ind_res]:
        par_filename = 'temp_fit_pars_'+M+'.npy'
        np.save(par_filename,parameters)
        ps.ind_res += 1
        if plot:
            plt.plot(k2,energies_computed[0],'r-')
            plt.plot(k2,energies_computed[1],'r-')
            plt.plot(k2,input_energy[0],'g*')
            plt.plot(k2,input_energy[1],'g*')
            plt.show()
    return res
#
def energies(parameters,M,a_mono,k_pts):
    hopping = ps.find_t(parameters)
    epsilon = ps.find_e(parameters)
    HSO = ps.find_HSO(parameters)
    ens = np.zeros((2,len(k_pts)))
    for i in range(len(k_pts)):
        K = k_pts[i]                                 #Considered K-point
        H_mono = H_monolayer(K,hopping,epsilon,HSO,a_mono)     #Compute UL Hamiltonian for given K
        temp = la.eigvalsh(H_mono)
        ens[0,i] = temp[13] + parameters[-1]     #offset in energy
        ens[1,i] = temp[12] + parameters[-1]     #offset in energy
    return ens
#

a_1 = np.array([1,0])
a_2 = np.array([-1/2,np.sqrt(3)/2])
J_plus = ((3,5), (6,8), (9,11))
J_minus = ((1,2), (3,4), (4,5), (6,7), (7,8), (9,10), (10,11))
J_MX_plus = ((3,1), (5,1), (4,2), (10,6), (9,7), (11,7), (10,8))
J_MX_minus = ((4,1), (3,2), (5,2), (9,6), (11,6), (10,7), (9,8), (11,8))
#####
#####Hamiltonian part
#####
#Here I contruct the single 11 orbital hamiltonian, which is 22x22 for spin-orbit, as a function of momentum
#Detail can be found in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.205108
def H_monolayer(K_p,hopping,epsilon,HSO,a_mono):
    t = hopping
    k_x,k_y = K_p       #momentum
    delta = a_mono* np.array([a_1, a_1+a_2, a_2, -(2*a_1+a_2)/3, (a_1+2*a_2)/3, (a_1-a_2)/3, -2*(a_1+2*a_2)/3, 2*(2*a_1+a_2)/3, 2*(a_2-a_1)/3])
    H_0 = np.zeros((11,11),dtype=complex)       #fist part without SO
    #Diagonal
    for i in range(11):
        H_0[i,i] += (epsilon[i] + 2*t[0][i,i]*np.cos(np.dot(K_p,delta[0])) 
                             + 2*t[1][i,i]*(np.cos(np.dot(K_p,delta[1])) + np.cos(np.dot(K_p,delta[2])))
                 )
    #Off diagonal symmetry +
    for ind in J_plus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (2*t[0][i,j]*np.cos(np.dot(K_p,delta[0])) 
                + t[1][i,j]*(np.exp(-1j*np.dot(K_p,delta[1])) + np.exp(-1j*np.dot(K_p,delta[2])))
                + t[2][i,j]*(np.exp(1j*np.dot(K_p,delta[1])) + np.exp(1j*np.dot(K_p,delta[2])))
                )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #Off diagonal symmetry -
    for ind in J_minus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (-2*1j*t[0][i,j]*np.sin(np.dot(K_p,delta[0])) 
                + t[1][i,j]*(np.exp(-1j*np.dot(K_p,delta[1])) - np.exp(-1j*np.dot(K_p,delta[2])))
                + t[2][i,j]*(-np.exp(1j*np.dot(K_p,delta[1])) + np.exp(1j*np.dot(K_p,delta[2])))
                )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #M-X coupling +
    for ind in J_MX_plus:
        i = ind[0]-1
        j = ind[1]-1
        temp = t[3][i,j] * (np.exp(1j*np.dot(K_p,delta[3])) - np.exp(1j*np.dot(K_p,delta[5])))
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #M-X coupling -
    for ind in J_MX_minus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (t[3][i,j] * (np.exp(1j*np.dot(K_p,delta[3])) + np.exp(1j*np.dot(K_p,delta[5])))
                   + t[4][i,j] * np.exp(1j*np.dot(K_p,delta[4]))
                   )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #Second nearest neighbor
    H_1 = np.zeros((11,11),dtype=complex)       #fist part without SO
    H_1[8,5] += t[5][8,5]*(np.exp(1j*np.dot(K_p,delta[6])) + np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,8] += np.conjugate(H_1[8,5])
    #
    H_1[10,5] += t[5][10,5]*(np.exp(1j*np.dot(K_p,delta[6])) - 1/2*np.exp(1j*np.dot(K_p,delta[7])) - 1/2*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,10] += np.conjugate(H_1[10,5])
    #
    H_1[9,5] += np.sqrt(3)/2*t[5][10,5]*(- np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,9] += np.conjugate(H_1[9,5])
    #
    H_1[8,7] += t[5][8,7]*(np.exp(1j*np.dot(K_p,delta[6])) - 1/2*np.exp(1j*np.dot(K_p,delta[7])) - 1/2*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[7,8] += np.conjugate(H_1[8,7])
    #
    H_1[8,6] += np.sqrt(3)/2*t[5][8,7]*(- np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,8] += np.conjugate(H_1[8,6])
    #
    H_1[9,6] += 3/4*t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,9] += np.conjugate(H_1[9,6])
    #
    H_1[10,6] += np.sqrt(3)/4*t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[7])) - np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,10] += np.conjugate(H_1[10,6])
    H_1[9,7] += H_1[10,6]
    H_1[7,9] += H_1[6,10]
    #
    H_1[10,7] += t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[6])) + 1/4*np.exp(1j*np.dot(K_p,delta[7])) + 1/4*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[7,10] += np.conjugate(H_1[10,7])
    #Combine the two terms
    H_TB = H_0 + H_1

    #### Spin orbit terms
    H = np.zeros((22,22),dtype = complex)
    H[:11,:11] = H_TB
    H[11:,11:] = H_TB
    H += HSO
    return H








