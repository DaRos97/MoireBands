import numpy as np
import matplotlib.pyplot as plt
import parameters as ps
import scipy.linalg as la
from time import time as tt

def convert(P,M):#input_filename):
    #for given path and material, takes the two bands and returns the lists of energy and momentum for the 2 top valence bands. 
    #For each k, is checked that the same k is also in the other band (there could be NAN)
    lines = []
    for i in range(2):
        input_filename = 'input_data/'+P+'_'+M+'_band'+str(i+1)+'_v1.txt'
        with open(input_filename, 'r') as f:
            lines.append(f.readlines())
    N = max(len(lines[0]),len(lines[1]))
    res1 = []
    res2 = []
    for i in range(N):
        try:
            temp1 = lines[0][i].split('\t')
            pas_1 = True
            if temp1[1] == 'NAN\n':
                pas_1 = False
        except:
            pas_1 = False
        try:
            temp2 = lines[1][i].split('\t')
            pas_2 = True
            if temp2[1] == 'NAN\n':
                pas_2 = False
        except:
            pas_2 = False
        if pas_1 and pas_2:
            res1.append([0,0])
            res1[-1][0] = float(temp1[0])
            res1[-1][1] = float(temp1[1])
            res2.append([0,0])
            res2[-1][0] = float(temp2[0])
            res2[-1][1] = float(temp2[1])
    res1 = np.array(res1)
    res2 = np.array(res2)
    return [res1,res2]

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

def find_vec_k(k_scalar,path,a_mono):
    #a1 = (1,0)
    #a2 = (-1/2,sqrt(3)/2)
    #b1 = (2\pi/a,2\pi/(sqrt(3)a))
    #b2 = (0,4\pi/(sqrt(3)a))
    #K = (4\pi/(3a),0)
    #K' = (2\pi/(3a),2\pi/(sqrt(3)a))
    #M = (\pi/a,\pi/(sqrt(3)a))
    k_pts = np.ndarray((len(k_scalar),2))
    if path == 'KGK':
        for i in range(len(k_scalar)):
            if k_scalar[i] < 0:
                k_pts[i,0] = np.abs(k_scalar[i])
                k_pts[i,1] = 0
            else:
                k_pts[i,0] = np.abs(k_scalar[i])*np.cos(np.pi/3)
                k_pts[i,1] = np.abs(k_scalar[i])*np.sin(np.pi/3)
    elif path == 'KMKp':
        M = np.array([np.pi,np.pi/np.sqrt(3)])/a_mono
        K = np.array([4*np.pi/3,0])/a_mono
        Kp = np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])/a_mono
        for i in range(len(k_scalar)):
            if k_scalar[i] < 0:
                k_pts[i,:] = M + (M-K)*np.abs(k_scalar[i])/la.norm(M-K)
            else:
                k_pts[i,:] = M + (M-Kp)*np.abs(k_scalar[i])/la.norm(M-Kp)
    else:
        print("Path not implemented in vectorization of k points")
        exit()
    return k_pts

def dft_values(pars,consider_SO):
    if consider_SO:
        return pars
    pars2 = pars[:40]
    pars2.append(pars[-1])
    return pars2

def chi2(pars,*args):
    input_energy, M, a_mono, N, k_pts, SO_pars = args
    if SO_pars == [0,0]:
        parameters = pars
        txt_SO = "SO"
    else:
        parameters = list(pars[:-1])
        parameters.append(SO_pars[0])
        parameters.append(SO_pars[1])
        parameters.append(pars[-1])
        txt_SO = "noSO"
    energies_computed = energies(parameters,M,a_mono,k_pts)
    res = 0
    for cut in range(len(k_pts)):
        for band in range(2):
            for i in range(N[cut]):
                res += (energies_computed[cut][band,i] - input_energy[cut][band][i])**2
        res /= np.sqrt(N[cut])
    if res < ps.list_res_bm[ps.ind_res]:
        par_filename = 'temp_fit_pars_'+M+'_'+txt_SO+'.npy'
        np.save(par_filename,parameters)
        ps.ind_res += 1
    return res
#
def energies(parameters,M,a_mono,k_pts):
    hopping = ps.find_t(parameters)
    epsilon = ps.find_e(parameters)
    HSO = ps.find_HSO(parameters)
    ens_tot = []
    for cut in range(len(k_pts)):
        ens = np.zeros((2,len(k_pts[cut])))
        for i in range(len(k_pts[cut])):
            K = k_pts[cut][i]                                 #Considered K-point
            H_mono = H_monolayer(K,hopping,epsilon,HSO,a_mono)     #Compute UL Hamiltonian for given K
            temp = la.eigvalsh(H_mono)
            ens[0,i] = temp[13] + parameters[-1]     #offset in energy
            ens[1,i] = temp[12] + parameters[-1]     #offset in energy
        ens_tot.append(ens)
    return ens_tot
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





####################################################

def symmetrize_data(input_data):
    N = len(input_data[0][:,0])
    cnt = 0
    for i in range(N):
        if input_data[0][i,0] < 0:
            cnt += 1
            continue
        break
    cnt -= 1
    res1 = np.ndarray((2*cnt,2))
    for i in range(cnt):
        res1[i,:] = input_data[0][i,:]
    for i in range(cnt):
        res1[cnt+i,0] = -input_data[0][cnt-i,0]
        res1[cnt+i,1] = input_data[0][cnt-i,1]
    res2 = np.ndarray((2*cnt,2))
    for i in range(cnt):
        res2[i,:] = input_data[1][i,:]
    for i in range(cnt):
        res2[cnt+i,0] = -input_data[1][cnt-i,0]
        res2[cnt+i,1] = input_data[1][cnt-i,1]
    return [res1, res2]






