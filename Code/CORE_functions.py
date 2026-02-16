import numpy as np
import scipy.linalg as la
from pathlib import Path

"""Functions related to Monolayer Hamiltonian"""

a_1 = np.array([1,0])
a_2 = np.array([-1/2,np.sqrt(3)/2])
J_plus = ((3,5), (6,8), (9,11))
J_minus = ((1,2), (3,4), (4,5), (6,7), (7,8), (9,10), (10,11))
J_MX_plus = ((3,1), (5,1), (4,2), (10,6), (9,7), (11,7), (10,8))
J_MX_minus = ((4,1), (3,2), (5,2), (9,6), (11,6), (10,7), (9,8), (11,8))

TMDs = ['WSe2','WS2']
m_list = [[-1,1],[-1,0],[0,-1],[1,-1],[1,0],[0,1]]  #for computing mini-BZ hoppings in moirè potential

def energy(parameters,HSO,data,TMD,bands=[],conduction=False):
    """Compute energy along the two cuts of 2 TopValenceBand for all considered k.

    """
    hopping = find_t(parameters)
    epsilon = find_e(parameters)
    a_TMD = dic_params_a_mono[TMD]
    offset = parameters[-3]
    #
    args_H = (hopping,epsilon,HSO,a_TMD,offset) #
    kpts = data.shape[0]
    all_H = H_monolayer(np.array(data[:,1:3]),*args_H)
    if len(bands)==0:
        nbands = 6# if TMD=='WSe2' else 2
    else:
        nbands = len(bands)
    ens = np.zeros((nbands,kpts))
    ensCond = np.zeros(kpts)        # Conduction band
    for i in range(kpts):
        #index of TVB is 13, the other is 12 (out of 22: 11 bands times 2 for SOC. 7/11 are valence -> 14 is the TVB)
        energies = la.eigvalsh(all_H[i])#[14-nbands:14][::-1]
        if len(bands)==0:
            ens[:,i] = energies[14-nbands:14][::-1]
        else:
            ens[:,i] = energies[bands][::-1]
        ensCond[i] = energies[14]
    if conduction:
        return ens, ensCond
    else:
        return ens

def H_monolayer(K_p,*args):
    """Monolayer Hamiltonian.
    Basis: 0-10 -> spin up, 11-21 -> spin down,
        0-10 -> [d_xz, d_yz,p_z^o,p_x^o,p_y^o,              #odd
                d_z2,d_xy,d_x2-y2,p_z^e,p_x^e,p_y^e]        #even

    """
    t,epsilon,HSO,a_mono,offset = args
    delta = a_mono* np.array([a_1, a_1+a_2, a_2, -(2*a_1+a_2)/3, (a_1+2*a_2)/3, (a_1-a_2)/3, -2*(a_1+2*a_2)/3, 2*(2*a_1+a_2)/3, 2*(a_2-a_1)/3])
    #First part without SO
    vec = True if len(K_p.shape)==2 else False
    H_0 = np.zeros((11,11),dtype=complex) if not vec else np.zeros((11,11,K_p.shape[0]),dtype=complex)
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
    H_1 = np.zeros((11,11),dtype=complex) if not vec else np.zeros((11,11,K_p.shape[0]),dtype=complex)
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
    H = np.zeros((22,22),dtype=complex) if not vec else np.zeros((22,22,K_p.shape[0]),dtype=complex)
    H[:11,:11] = H_TB
    H[11:,11:] = H_TB
    #
    if len(H.shape)==3:
        H = np.transpose(H,(2,0,1))
    H += HSO
    #Offset
    H += np.identity(22)*offset
    return H

def find_t(dic_params_H):
    """Define hopping matrix elements from inputs and complete all symmetry related ones.

    """
    t = []
    t.append(np.zeros((11,11))) #t1
    t.append(np.zeros((11,11))) #t2
    t.append(np.zeros((11,11))) #t3
    t.append(np.zeros((11,11))) #t4
    t.append(np.zeros((11,11))) #t5
    t.append(np.zeros((11,11))) #t6
    #Independent parameters
    t[0][0,0] = dic_params_H[7]
    t[0][1,1] = dic_params_H[8]
    t[0][2,2] = dic_params_H[9]
    t[0][3,3] = dic_params_H[10]
    t[0][4,4] = dic_params_H[11]
    t[0][5,5] = dic_params_H[12]
    t[0][6,6] = dic_params_H[13]
    t[0][7,7] = dic_params_H[14]
    t[0][8,8] = dic_params_H[15]
    t[0][9,9] = dic_params_H[16]
    t[0][10,10] = dic_params_H[17]
    t[0][2,4] = dic_params_H[18]
    t[0][5,7] = dic_params_H[19]
    t[0][8,10] = dic_params_H[20]
    t[0][0,1] = dic_params_H[21]
    t[0][2,3] = dic_params_H[22]
    t[0][3,4] = dic_params_H[23]
    t[0][5,6] = dic_params_H[24]
    t[0][6,7] = dic_params_H[25]
    t[0][8,9] = dic_params_H[26]
    t[0][9,10] = dic_params_H[27]
    t[4][3,0] = dic_params_H[28]
    t[4][2,1] = dic_params_H[29]
    t[4][4,1] = dic_params_H[30]
    t[4][8,5] = dic_params_H[31]
    t[4][10,5] = dic_params_H[32]
    t[4][9,6] = dic_params_H[33]
    t[4][8,7] = dic_params_H[34]
    t[4][10,7] = dic_params_H[35]
    t[5][8,5] = dic_params_H[36]
    t[5][10,5] = dic_params_H[37]
    t[5][8,7] = dic_params_H[38]
    t[5][10,7] = dic_params_H[39]
    #Non-independent parameters
    list_1 = ((1,2,-1),(4,5,3),(7,8,6),(10,11,9))
    for inds in list_1:
        a,b,g = inds
        a -= 1
        b -= 1
        g -= 1
        t[1][a,a] = 1/4*t[0][a,a] + 3/4*t[0][b,b]
        t[1][b,b] = 3/4*t[0][a,a] + 1/4*t[0][b,b]
        t[1][a,b] = np.sqrt(3)/4*(t[0][a,a]-t[0][b,b]) - t[0][a,b]
        t[2][a,b] = -np.sqrt(3)/4*(t[0][a,a]-t[0][b,b]) - t[0][a,b]
        if g >= 0:
            t[1][g,g] = t[0][g,g]
            t[1][g,b] = np.sqrt(3)/2*t[0][g,a]-1/2*t[0][g,b]
            t[2][g,b] = -np.sqrt(3)/2*t[0][g,a]-1/2*t[0][g,b]
            t[1][g,a] = np.sqrt(3)/2*t[0][g,b]+1/2*t[0][g,a]
            t[2][g,a] = -np.sqrt(3)/2*t[0][g,b]+1/2*t[0][g,a]
    list_2 = ((1,2,4,5,3),(7,8,10,11,9))
    for inds in list_2:
        a,b,ap,bp,gp = inds
        a -= 1
        b -= 1
        ap -= 1
        bp -= 1
        gp -= 1
        t[3][ap,a] = 1/4*t[4][ap,a] + 3/4*t[4][bp,b]
        t[3][bp,b] = 3/4*t[4][ap,a] + 1/4*t[4][bp,b]
        t[3][bp,a] = t[3][ap,b] = -np.sqrt(3)/4*t[4][ap,a] + np.sqrt(3)/4*t[4][bp,b]
        t[3][gp,a] = -np.sqrt(3)/2*t[4][gp,b]
        t[3][gp,b] = -1/2*t[4][gp,b]
    t[3][8,5] = t[4][8,5]
    t[3][9,5] = -np.sqrt(3)/2*t[4][10,5]
    t[3][10,5] = -1/2*t[4][10,5]
    return t

def find_e(dic_params_H):
    """Define the array of on-site energies from inputs and symmetry related ones.

    """
    e = np.zeros(11)
    e[0] = dic_params_H[0]
    e[1] = e[0]
    e[2] = dic_params_H[1]
    e[3] = dic_params_H[2]
    e[4] = e[3]
    e[5] = dic_params_H[3]
    e[6] = dic_params_H[4]
    e[7] = e[6]
    e[8] = dic_params_H[5]
    e[9] = dic_params_H[6]
    e[10] = e[9]
    return e

def find_HSO(SO_pars):
    """Compute the SO Hamiltonian. Taken from arXiv:1401....(paco guinea)
    They compute exactly the same thing BUT with a basis change.

    """
    if len(SO_pars)>2:
        print("Wrong pars in HSO")
        exit()
    l_M = SO_pars[0]
    l_X = SO_pars[1]
    ####
    Mee_uu = np.zeros((6,6),dtype=complex)
    Mee_uu[1,2] = -1j*l_M
    Mee_uu[2,1] = 1j*l_M
    #
    Mee_uu[3,4] = -1j*l_X/2
    Mee_uu[4,3] = 1j*l_X/2
    Mee_dd = -Mee_uu
    ###
    Moo_uu = np.zeros((5,5),dtype=complex)
    Moo_uu[0,1] = -1j*l_M/2
    Moo_uu[1,0] = 1j*l_M/2
    #
    Moo_uu[2,3] = -1j*l_X/2
    Moo_uu[3,2] = 1j*l_X/2
    Moo_dd = -Moo_uu
    ###
    Meo_ud = np.zeros((6,5),dtype=complex)
    Meo_ud[0,0] = -l_M*np.sqrt(3)/2
    Meo_ud[0,1] = 1j*l_M*np.sqrt(3)/2
    Meo_ud[1,0] = l_M/2
    Meo_ud[1,1] = 1j*l_M/2
    Meo_ud[2,0] = -1j*l_M/2
    Meo_ud[2,1] = l_M/2
    #
    Meo_ud[3,4] = l_X/2
    Meo_ud[4,4] = -1j*l_X/2
    Meo_ud[5,2] = -l_X/2
    Meo_ud[5,3] = 1j*l_X/2
    Moe_du = np.conjugate(Meo_ud.T)
    ###
    Meo_du = np.zeros((6,5),dtype=complex)
    Meo_du[0,0] = l_M*np.sqrt(3)/2
    Meo_du[0,1] = 1j*np.sqrt(3)*l_M/2
    Meo_du[1,0] = -l_M/2
    Meo_du[1,1] = 1j*l_M/2
    Meo_du[2,0] = -1j*l_M/2
    Meo_du[2,1] = -l_M/2
    #
    Meo_du[3,4] = -l_X/2
    Meo_du[4,4] = -1j*l_X/2
    Meo_du[5,2] = l_X/2
    Meo_du[5,3] = 1j*l_X/2
    Moe_ud = np.conjugate(Meo_du.T)
    ###
    Muu = np.zeros((11,11),dtype=complex)
    Muu[:6,:6] = Mee_uu
    Muu[6:,6:] = Moo_uu
    Mdd = np.zeros((11,11),dtype=complex)
    Mdd[:6,:6] = Mee_dd
    Mdd[6:,6:] = Moo_dd
    Mud = np.zeros((11,11),dtype=complex)
    Mud[:6,6:] = Meo_ud
    Mud[6:,:6] = Moe_ud
    Mdu = np.zeros((11,11),dtype=complex)
    Mdu[:6,6:] = Meo_du
    Mdu[6:,:6] = Moe_du
    #
    HSO = np.zeros((22,22),dtype=complex)
    HSO[:11,:11] = Muu
    HSO[11:,11:] = Mdd
    HSO[:11,11:] = Mud
    HSO[11:,:11] = Mdu
    #### Now we do the basis transformation
    #Basis now: (d_z2,d_x2-y2,d_xy,px_s,py_s,pz_a,d_xz,d_yz,px_a,py_a,pz_s)
    #           (a,b,c,d,e,f,g,h,i,j,k)
    #Basis after: (d_xz,d_yz,pz_s,px_a,py_a,d_z2,d_xy,d_x2-y2,pz_a,px_s,py_s)
    #           (6,7,10,8,9,0,2,1,5,3,4)
    #Matrix transformation: P
    #   |0 0 0 0 0 1 0 0 0 0 0|
    #   |0 0 0 0 0 0 0 1 0 0 0|
    #   |0 0 0 0 0 0 1 0 0 0 0|
    #   |0 0 0 0 0 0 0 0 0 1 0|
    #   |0 0 0 0 0 0 0 0 0 0 1|
    #   |0 0 0 0 0 0 0 0 1 0 0|
    #   |1 0 0 0 0 0 0 0 0 0 0|
    #   |0 1 0 0 0 0 0 0 0 0 0|
    #   |0 0 0 1 0 0 0 0 0 0 0|
    #   |0 0 0 0 1 0 0 0 0 0 0|
    #   |0 0 1 0 0 0 0 0 0 0 0|
    if 0:
        P = np.zeros((11,11))
        P[0,5] = P[1,7] = P[2,6] = P[3,9] = P[4,10] = P[5,8] = 1
        P[6,0] = P[7,1] = P[8,3] = P[9,4] = P[10,2] = 1
        Pf = np.zeros((22,22))
        Pf[:11,:11] = P
        Pf[11:,11:] = P
        HSOf = np.linalg.inv(Pf) @ HSO @ Pf
    #Just swap the columns and rows
    HSOf = HSO[[6,7,10,8,9,0,2,1,5,3,4,17,18,21,19,20,11,13,12,16,14,15],:]
    HSOf = HSOf[:,[6,7,10,8,9,0,2,1,5,3,4,17,18,21,19,20,11,13,12,16,14,15]]
    return HSOf

def get_kList(cut,kPts,TMD='WSe2',endpoint=False,returnNorm=False):
    """
    Get cut in BZ to compute.
    The cut is composed of a list of high symmetry points separated by '-', supported: Kp, K, M, G.
    The number of points in the list are divided in the different segments depending on their actual BZ length.
    """
    b2 = 4*np.pi/np.sqrt(3)/dic_params_a_mono[TMD] * np.array([0,1])
    b1 = R_z(-np.pi/3) @ b2
    b6 = R_z(-2*np.pi/3) @ b2
    K = (b1+b6)/3
    Kp = R_z(np.pi/3)@K
    G = np.array([0,0])
    M = b1/2
    dic_Kpts = {'K':K,'Kp':Kp,'G':G,'M':M}
    terms = cut.split('-')
    dks = np.zeros(len(terms)-1)    #compute absolute values of distances b/w points
    for i in range(len(terms)-1):
        dks[i] = np.linalg.norm(dic_Kpts[terms[i+1]]-dic_Kpts[terms[i]])
    tot_k = np.sum(dks)     #sum to have total length of path
    ls = np.zeros(len(terms)-1,dtype=int)   #define number of points for each segment
    for i in range(len(terms)-1):
        ls[i] = int(dks[i]/tot_k*kPts)
        if i==len(terms)-2 and endpoint:
            ls[i] += 1
    kPts = ls.sum()     #adjust total number of points
    res = np.zeros((kPts,2))
    for i in range(len(terms)-1):
        for p in range(ls[i]):
            ind = p if i==0 else p+ls[:i].sum()
            res[ind] = dic_Kpts[terms[i]] + (dic_Kpts[terms[i+1]] - dic_Kpts[terms[i]])/ls[i]*p
    if returnNorm:
        norm = np.zeros(kPts)
        for i in range(1,kPts):
            norm[i] = norm[i-1] + np.linalg.norm(res[i]-res[i-1])
        return res, norm
    else:
        return res

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

"""Utility functions"""

def getFilename(*args,dirname='',extension='',floatPrecision=6):
    """ Get filename for set of parameters.

    Parameters
    ----------
    *args: list of arguments to put in the filename.
    dirname: str, should end with '/'
    extension: string, should start with '.'
    floatPrecision: int, detail of floating point arguments.

    Returns
    -------
    filename : string.
    """
    if len(dirname)>0 and dirname[-1]!='/':
        raise ValueError("directory name  %s must end with '/'"%dirname)
    if len(extension)>0 and extension[0]!='.':
        raise ValueError("extension name  %s must begin with '.'"%extension)

    filename = ''
    filename += dirname
    for i,a in enumerate(args):
        t = type(a)
        if t in [str,np.str_]:
            filename += a
        elif t in [int, np.int64, np.int32]:
            filename += str(a)
        elif t in [float, np.float32, np.float64]:
            filename += f"{a:.{floatPrecision}f}"
        elif t==tuple:
            filename += getFilename(*a)
        else:
            raise TypeError("Parameter %s has unsupported type: %s"%(a,t))
        if not i==len(args)-1:
            filename += '_'
    filename += extension
    return filename

def get_machine(cwd):
    """Selects the machine the code is running on by looking at the working directory. Supports local, hpc (baobab or yggdrasil) and mafalda.

    Parameters
    ----------
    pwd : string
        Result of os.pwd(), the working directory.

    Returns
    -------
    string
        An acronim for the computing machine.
    """
    if cwd[6:11] == 'dario':
        return 'loc'
    elif cwd[:20] == '/home/users/r/rossid':
        return 'hpc'
    elif cwd[:13] == '/users/rossid':
        return 'maf'

def R_z(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

def tqdm(x,**kwargs):
    return x


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

"""DFT parameters and constants"""

dic_params_a_mono = {
    'WS2': 3.18,
    'WSe2': 3.32,
    }

params_V = [0.0335,np.pi, 7.7*1e-3, -106*2*np.pi/360]

"""Twist angle of samples 3 and 11. Central value and error bars."""
dic_params_twist = {
        'S3': [1.5,1.8,2.1],
        'S11': [2.6,2.8,3.0],
        }

def moire_length(theta):
    """
    Real space moire lattice length.
    """
    return 1/np.sqrt(1/dic_params_a_mono['WSe2']**2+1/dic_params_a_mono['WS2']**2-2*np.cos(theta)/dic_params_a_mono['WSe2']/dic_params_a_mono['WS2'])

def miniBZ_rotation(theta):
    """
    Compute eta, the rotation of the mini-BZ wrt the monolayer BZ.
    It's the angle of the segment between the K points of the 2 layers.
    """
    return np.arctan(np.tan(theta/2)*(dic_params_a_mono['WSe2']+dic_params_a_mono['WS2'])/(dic_params_a_mono['WSe2']-dic_params_a_mono['WS2']))

def get_reciprocal_moire(theta):
    """
    Compute moire reciprocal lattice vectors.
    They depend on the moire length for the size and on the orientation of the mini-BZ for the direction.
    Returns a list of 7 vectors, first one being 0.
    """
    eta = miniBZ_rotation(theta)
    Mat = R_z(eta)
    G1 = 4*np.pi/np.sqrt(3)/moire_length(theta)*np.array([np.sqrt(3)/2,1/2])
    G_M = [np.zeros(2), Mat@G1,]
    for i in range(1,6):
        G_M.append(R_z(np.pi/3*i) @ G_M[1])
    return G_M

def get_lattice_vectors(TMD,theta=0):
    """
    Compute single layer real-space and reciprocal lattice vectors.
    The conventions we use are:
        - WSe2 -> unit cell is aligned, with vertical bonds.
        - WS2 -> unit cell has theta rotation.
    """
    As = [ R_z(theta/180*np.pi) @ a_1*dic_params_a_mono[TMD], ]
    for i in range(1,6):
        As.append(R_z(np.pi/3*i) @ As[0])
    # Compute area (scalar cross product)
    area = As[0][0]*As[1][1] - As[0][1]*As[1][0]
    # Reciprocal lattice vectors
    Bs = [ 2 * np.pi * np.array([ As[1][1], -As[1][0]]) / area, ]
    for i in range(1,6):
        Bs.append(R_z(np.pi/3*i) @ Bs[0])
    # Rotate Bs to stick with conventions
    Bs = Bs[1:] + Bs[:1]
    return As, Bs


initial_pt = {
        'WS2': [
            #'e1':   
            1.3754,
            #'e3':   
            -1.1278,
            #'e4':   
            -1.5534,
            #'e6':   
            -0.0393,
            #'e7':   
            0.1984,
            #'e9':   
            -3.3706,
            #'e10':  
            -2.3461,

            #'t1_11':   
            -0.2011,
            #'t1_22':   
            0.0263,
            #'t1_33':   
            -0.1749,
            #'t1_44':   
            0.8726,
            #'t1_55':   
            -0.2187,
            #'t1_66':   
            -0.3716,
            #'t1_77':   
            0.3537,
            #'t1_88':   
            -0.6892,
            #'t1_99':   
            -0.2112,
            #'t1_1010':   
            0.9673,
            #'t1_1111':     17 
            0.0143,
            #'t1_35':   
            -0.0818,
            #'t1_68':   
            0.4896,
            #'t1_911':   
            -0.0315,
            #'t1_12':   
            -0.3106,
            #'t1_34':   
            -0.1105,
            #'t1_45':   
            -0.0989,
            #'t1_67':   
            -0.1467,
            #'t1_78':   
            -0.3030,
            #'t1_910':   
            0.1645,
            #'t1_1011':   
            -0.1018,

            #'t5_41':   
            -0.8855,
            #'t5_32':   
            -1.4376,
            #'t5_52':   
            2.3121,
            #'t5_96':   
            -1.0130,
            #'t5_116':   
            -0.9878,
            #'t5_107':   
            1.5629,
            #'t5_98':   
            -0.9491,
            #'t5_118':   
            0.6718,

            #'t6_96':   
            -0.0659,
            #'t6_116':   
            -0.1533,
            #'t6_98':   
            -0.2618,
            #'t6_118':          39 
            -0.2736,

            #'offset
            -1.350,

            #SO
            #'W':
            0.2874,
            #'S'
            0.0556,

                ],
        'WSe2': [
            #'e1':   
            1.0349,
            #'e3':   
            -0.9573,
            #'e4':   
            -1.3937,
            #'e6':   
            -0.1667,
            #'e7':   
            0.0984,
            #'e9':   
            -3.3642,
            #'e10':   
            -2.1820,

            #'t1_11':   
            -0.1395,
            #'t1_22':   
            0.0129,
            #'t1_33':   
            -0.2171,
            #'t1_44':   
            0.9763,
            #'t1_55':   
            -0.1985,
            #'t1_66':   
            -0.3330,
            #'t1_77':   
            0.3190,
            #'t1_88':   
            -0.5837,
            #'t1_99':   
            -0.2399,
            #'t1_1010':   
            1.0470,
            #'t1_1111':   
            0.0029,
            #'t1_35':   
            -0.0912,
            #'t1_68':   
            0.4233,
            #'t1_911':   
            -0.0377,
            #'t1_12':   
            -0.2321,
            #'t1_34':   
            -0.0797,
            #'t1_45':   
            -0.0920,
            #'t1_67':   
            -0.1250,
            #'t1_78':   
            -0.2456,
            #'t1_910':   
            0.1857,
            #'t1_1011':   
            -0.1027,

            #'t5_41':   
            -0.7744,
            #'t5_32':   
            -1.4014,
            #'t5_52':   
            2.0858,
            #'t5_96':   
            -0.8998,
            #'t5_116':   
            -0.9044,
            #'t5_107':   
            1.4030,
            #'t5_98':   
            -0.8548,
            #'t5_118':   
            0.5711,

            #'t6_96':   
            -0.0676,
            #'t6_116':   
            -0.1608,
            #'t6_98':   
            -0.2618,
            #'t6_118':   
            -0.2424,

            #'offset
            -0.736,

            #SO
            #'W':
            0.2874,
            #'Se'
            0.2470,
                ],
        }

#Names of independent parameters of the model
list_names_all = [
            'e1',
            'e3',
            'e4',
            'e6',
            'e7',
            'e9',
            'e10',
            't1_11',
            't1_22',
            't1_33',
            't1_44',
            't1_55',
            't1_66',
            't1_77',
            't1_88',
            't1_99',
            't1_1010',
            't1_1111',
            't1_35',
            't1_68',
            't1_911',
            't1_12',
            't1_34',
            't1_45',
            't1_67',
            't1_78',
            't1_910',
            't1_1011',
            't5_41',
            't5_32',
            't5_52',
            't5_96',
            't5_116',
            't5_107',
            't5_98',
            't5_118',
            't6_96',
            't6_116',
            't6_98',
            't6_118',
            'offset',
            'L_W',
            'L_S',
            ]
#Formatted names of independent parameters of the model
list_formatted_names_all = [
            r'$\epsilon_1$',
            r'$\epsilon_3$',
            r'$\epsilon_4$',
            r'$\epsilon_6$',
            r'$\epsilon_7$',
            r'$\epsilon_9$',
            r'$\epsilon_{10}$',
            '$t^{(1)}_{1,1}$',
            '$t^{(1)}_{2,2}$',
            '$t^{(1)}_{3,3}$',
            '$t^{(1)}_{4,4}$',
            '$t^{(1)}_{5,5}$',
            '$t^{(1)}_{6,6}$',
            '$t^{(1)}_{7,7}$',
            '$t^{(1)}_{8,8}$',
            '$t^{(1)}_{9,9}$',
            '$t^{(1)}_{10,10}$',
            '$t^{(1)}_{11,11}$',
            '$t^{(1)}_{3,5}$',
            '$t^{(1)}_{6,8}$',
            '$t^{(1)}_{9,11}$',
            '$t^{(1)}_{1,2}$',
            '$t^{(1)}_{3,4}$',
            '$t^{(1)}_{4,5}$',
            '$t^{(1)}_{6,7}$',
            '$t^{(1)}_{7,8}$',
            '$t^{(1)}_{9,10}$,',
            '$t^{(1)}_{10,11}$',
            '$t^{(5)}_{4,1}$',
            '$t^{(5)}_{3,2}$',
            '$t^{(5)}_{5,2}$',
            '$t^{(5)}_{9,6}$',
            '$t^{(5)}_{11,6}$',
            '$t^{(5)}_{10,7}$',
            '$t^{(5)}_{9,8}$',
            '$t^{(5)}_{11,8}$',
            '$t^{(6)}_{9,6}$',
            '$t^{(6)}_{11,6}$',
            '$t^{(6)}_{9,8}$',
            '$t^{(6)}_{11,8}$',
            '$offset$',
            r'$\lambda_W$',
            r'$\lambda_{Se}$',
            ]

"""Parameters of experimental image

In order, are:
    E_max, E_min,
    pixel of k=-1, pixel of k=1, pixel of e=E_max, pixel of e=E_min

"""

dic_pars_samples = {
        'S11': [0.0, -3.5,       #adjusted to have same scale as S3
                810, 2371, 89, 1899],
        'S3':  [0, -2.5,
                697, 2156, 108, 1681],
        'S11zoom': [-0.6, -1.8,
                840, 2980, 86, 1147],
        }

dic_energy_bounds = {'S11zoom':(-0.6,-1.8), 'S11':(-0.5,-2.5), 'S3':(-0.2,-1.8)}

#Interlayer parameters of "constant" part w1 for p and d orbitals in the different cases
w1p_dic = {
    'DFT':{'S3':-1.650 , 'S11':-1.820},
    'fit':{'S3':-1.725 , 'S11':-1.725}#-1.92}
          }
w1d_dic = {
    'DFT':{'S3':0.340 , 'S11':0.420},
    'fit':{'S3':0.370 , 'S11':0.370}#0.46}
          }



class monolayerData():
    def __init__(self,TMD):
        self.TMD = TMD
        self.paths = ['KGK','KMKp']
        self.nbands = {'WSe2':{'KGK':6,'KMKp':4},'WS2':{'KGK':6,'KMKp':4}}[TMD]
        self.raw_data = self._getRaw()
        self.sym_data = self._getSym()
        self.kpoints_sym = self._getKpoints()
        self.offset = {'WSe2':-0.052,'WS2':0.01}

    def _getRaw(self):
        """ Raw data comes from:
            'Inputs/' folder for the first two bands of KGK
            'Inputs/fitM/' folder for the KMKp path -> 2 bands close to K and 4 bands close to M
            'Inputs/fitGammaLower/' folder for the additional 4 bands below G in the KGK path -> these are ony for negative momenta so no need to symmetrize them.
        """
        raw = {}
        for path in self.paths:
            raw[path] = []
            nbands = self.nbands[path]
            if nbands==6:
                nbands = 2
            if nbands==4:
                nbands = 6
            for ib in range(nbands):
                if nbands==2:
                    fn = Path('Inputs/'+path+'_'+self.TMD+'_band%d'%(ib+1)+'.txt')
                else:
                    fn = Path('Inputs/fitM/%s_%s_band%d_v3.txt'%(path,self.TMD,ib+1))
                with open(fn,'r') as f:
                    lines = f.readlines()
                temp = []
                for il in range(len(lines)):
                    k,e = lines[il].split('\t')
                    if e=='NAN\n':
                        temp.append([float(k),np.nan])
                    else:
                        temp.append([float(k),float(e)])
                temp = np.array(temp)
                if ib in [0,1]:
                    raw[path].append(temp)
                if ib in [2,3]:
                    raw[path].append(temp)
                if ib in [4,5]:
                    oldb = 3 if ib==4 else 2
                    if self.TMD=='WS2':
                        temp = temp[::-1]
                    raw[path][oldb] = np.concatenate([raw[path][oldb],temp])
            if (self.TMD=='WSe2' and path=='KMKp'):
                raw[path][0], raw[path][1] = raw[path][1], raw[path][0]
            if path=='KGK':
                for ib in range(4):
                    fn = Path('Inputs/fitGammaLower/'+'KGK_%s_band%d'%(self.TMD,ib+1)+'.txt')
                    with open(fn,'r') as f:
                        lines = f.readlines()
                    temp = []
                    for il in range(len(lines)):
                        k,e = lines[il].split('\t')
                        #print(k,e)
                        if e=='NAN\n' or e=='\n':
                            temp.append([float(k),np.nan])
                        else:
                            temp.append([abs(float(k)),float(e)])
                    temp = np.array(temp)
                    raw[path].append(temp)
        return raw

    def _getSym(self):
        sym = {}
        for path in self.paths:
            sym[path] = []
            nbands = self.nbands[path]
            for ib in range(nbands):
                rd = self.raw_data[path][ib]
                if ib>=2 and path=='KGK':
                    sym[path].append(self.raw_data[path][ib])
                    continue
                nk = rd.shape[0]
                nkl = nk//2
                nkr = nk//2 if nk%2==0 else nk//2+1
                temp = np.zeros((nkl,2))
                temp[:,0] = rd[nkr:,0]
                rd_m = rd[:nkl,1][::-1]
                rd_p = rd[nkr:,1]
                mask_m = ~np.isnan(rd_m)
                mask_p = ~np.isnan(rd_p)
                mask_tot = mask_m & mask_p
                #
                temp[mask_tot,1] = (rd_m[mask_tot]+rd_p[mask_tot])/2
                mask_tm = mask_m & ~mask_p
                temp[mask_tm,1] = rd_m[mask_tm]
                mask_tp = ~mask_m & mask_p
                temp[mask_tp,1] = rd_p[mask_tp]
                temp = np.delete(temp, ~mask_m & ~mask_p, axis=0)
                if nk%2==1:
                    temp = np.insert(temp, 0, rd[nk//2], axis=0)
                sym[path].append(temp)
        return sym

    def _getKpoints(self):
        kpts = {}
        M = np.array([np.pi,np.pi/np.sqrt(3)])/dic_params_a_mono[self.TMD]
        K = np.array([4*np.pi/3,0])/dic_params_a_mono[self.TMD]
        Kp = np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])/dic_params_a_mono[self.TMD]
        for path in self.paths:
            kpts[path] = []
            nbands = self.nbands[path]
            for ib in range(nbands):
                sd = self.sym_data[path][ib]
                temp = np.zeros((sd.shape[0],2))
                if path=='KGK':
                    temp[:,0] = sd[:,0]
                else:
                    for ik in range(sd.shape[0]):
                        temp[ik] = M + (Kp-M)*sd[ik,0]/la.norm(Kp-M)
                kpts[path].append(temp)
        return kpts

class dataWS2(monolayerData):
    def __init__(self):
        super().__init__(TMD='WS2')

    def getFitData(self,ptsPerPath=(40,20,20)):
        """ Here we compute the final array to give for the fitting.
        It has (ptsPerPath * #paths) elements, each with 9 entries:
            - |k| (shfted by |K| for the KMKp path -> for plotting)
            - kx
            - ky
            - energy band 1
            - energy band 2
            - energy band 3
            - energy band 4
            - energy band 5
            - energy band 6
        Bands 3 and 4 are NAN except for |k| close to M and to Gamma.
        Bands 5 and 6 just close to Gamma.
        """
        M = np.array([np.pi,np.pi/np.sqrt(3)])/dic_params_a_mono[self.TMD]
        K = np.array([4*np.pi/3,0])/dic_params_a_mono[self.TMD]
        modM = la.norm(M-K)
        modK = la.norm(K)
        data = np.zeros((ptsPerPath[0]+ptsPerPath[1]+ptsPerPath[2],9))
        # KGK
        sd = self.sym_data[self.paths[0]]
        kk = self.kpoints_sym[self.paths[0]]
        kmin,kmax = (sd[0][0,0],sd[0][-1,0])
        kvals = np.linspace(kmin,kmax,ptsPerPath[0])
        data[:ptsPerPath[0],0] = kvals
        data[:ptsPerPath[0],1] = np.interp( kvals, sd[0][:,0], kk[0][:,0] )
        data[:ptsPerPath[0],2] = np.interp( kvals, sd[0][:,0], kk[0][:,1] )
        data[:ptsPerPath[0],3] = np.interp( kvals, sd[0][:,0], sd[0][:,1] )
        data[:ptsPerPath[0],4] = np.interp( kvals, sd[1][:,0], sd[1][:,1] )
        # bands 34
        mask34 = kvals <= sd[2][0,0]        #ordered backwards
        kvals34 = kvals[mask34]
        maskNan = ~np.isnan(sd[3][::-1,1])
        data[:kvals34.shape[0],5] = np.interp( kvals34, sd[3][::-1,0][maskNan], sd[3][::-1,1][maskNan] )
        data[:kvals34.shape[0],6] = np.interp( kvals34, sd[2][::-1,0], sd[2][::-1,1] )
        data[kvals34.shape[0]:ptsPerPath[0],5:7] = np.nan
        # bands 56
        mask56 = kvals <= sd[4][0,0]        #ordered backwards
        kvals56 = kvals[mask56]
        maskNan = ~np.isnan(sd[4][::-1,1])
        data[:kvals56.shape[0],7] = np.interp( kvals56, sd[5][::-1,0], sd[5][::-1,1] )
        data[:kvals56.shape[0],8] = np.interp( kvals56, sd[4][::-1,0][maskNan], sd[4][::-1,1][maskNan] )
        data[kvals56.shape[0]:ptsPerPath[0],7:9] = np.nan
        # KMKp
        sd = self.sym_data[self.paths[1]]
        kk = self.kpoints_sym[self.paths[1]]
        kmin,kmax = (sd[0][-1,0],sd[3][-1,0])
        kvals = np.linspace(kmin,kmax,ptsPerPath[1])
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],0] = modK + modM - kvals
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],1] = np.interp( kvals, sd[0][:,0], kk[0][:,0] )
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],2] = np.interp( kvals, sd[0][:,0], kk[0][:,1] )
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],3] = np.interp( kvals, sd[0][:,0], sd[0][:,1] )
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],4] = np.interp( kvals, sd[1][:,0], sd[1][:,1] )
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],5:9] = np.nan
        # KMKp close to M
        kmin,kmax = (sd[3][-1,0],sd[3][0,0])
        kvals = np.linspace(kmin,kmax,ptsPerPath[2])
        data[ptsPerPath[0]+ptsPerPath[1]:,0] = modK + modM - kvals
        data[ptsPerPath[0]+ptsPerPath[1]:,1] = np.interp( kvals, sd[0][:,0], kk[0][:,0] )
        data[ptsPerPath[0]+ptsPerPath[1]:,2] = np.interp( kvals, sd[0][:,0], kk[0][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,3] = np.interp( kvals, sd[0][:,0], sd[0][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,4] = np.interp( kvals, sd[1][:,0], sd[1][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,5] = np.interp( kvals, sd[2][:,0], sd[2][:,1] )
        data[-5:,5] = np.nan        #specific for (30,15,10)
        data[ptsPerPath[0]+ptsPerPath[1]:,6] = np.interp( kvals, sd[3][:,0], sd[3][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,7:9] = np.nan

        # Order bands of index 1 and 2 which shifts at some point
        pp = ptsPerPath[0]+ptsPerPath[1]
        mask = data[pp:,4]<data[pp:,5]
        data[pp:,4][mask], data[pp:,5][mask] = data[pp:,5][mask], data[pp:,4][mask]
        # Shift of KMKp
        data[ptsPerPath[0]:,3] += self.offset[self.TMD]
        data[ptsPerPath[0]:,4] += self.offset[self.TMD]
        data[ptsPerPath[0]:,5] += self.offset[self.TMD]
        data[ptsPerPath[0]:,6] += self.offset[self.TMD]
        return data

class dataWSe2(monolayerData):
    def __init__(self):
        super().__init__(TMD='WSe2')

    def getFitData(self,ptsPerPath=(20,20,20)):
        """ Here we compute the final array to give for the fitting.
        It has (ptsPerPath * #paths) elements, each with 9 entries:
            - |k| (shfted by |K| for the KMKp path -> for plotting)
            - kx
            - ky
            - energy band 1
            - energy band 2
            - energy band 3
            - energy band 4
            - energy band 5
            - energy band 6
        Bands 3 and 4 are NAN except for |k| close to M and to Gamma.
        Bands 5 and 6 just close to Gamma.
        """
        M = np.array([np.pi,np.pi/np.sqrt(3)])/dic_params_a_mono[self.TMD]
        K = np.array([4*np.pi/3,0])/dic_params_a_mono[self.TMD]
        modM = la.norm(M-K)
        modK = la.norm(K)
        data = np.zeros((ptsPerPath[0]+ptsPerPath[1]+ptsPerPath[2],9))
        # KGK
        sd = self.sym_data[self.paths[0]]
        kk = self.kpoints_sym[self.paths[0]]
        kmin,kmax = (sd[0][0,0],sd[0][-1,0])
        kvals = np.linspace(kmin,kmax,ptsPerPath[0])
        data[:ptsPerPath[0],0] = kvals
        data[:ptsPerPath[0],1] = np.interp( kvals, sd[0][:,0], kk[0][:,0] )
        data[:ptsPerPath[0],2] = np.interp( kvals, sd[0][:,0], kk[0][:,1] )
        data[:ptsPerPath[0],3] = np.interp( kvals, sd[0][:,0], sd[0][:,1] )
        data[:ptsPerPath[0],4] = np.interp( kvals, sd[1][:,0], sd[1][:,1] )
        # bands 34
        mask34 = kvals <= sd[2][0,0]        #ordered backwards
        kvals34 = kvals[mask34]
        maskNan = ~np.isnan(sd[3][::-1,1])
        data[:kvals34.shape[0],5] = np.interp( kvals34, sd[3][::-1,0][maskNan], sd[3][::-1,1][maskNan] )
        data[:kvals34.shape[0],6] = np.interp( kvals34, sd[2][::-1,0], sd[2][::-1,1] )
        data[kvals34.shape[0]:ptsPerPath[0],5:7] = np.nan
        # bands 56
        mask56 = kvals <= sd[4][0,0]        #ordered backwards
        kvals56 = kvals[mask56]
        maskNan = ~np.isnan(sd[4][::-1,1])
        data[:kvals56.shape[0],7] = np.interp( kvals56, sd[5][::-1,0], sd[5][::-1,1] )
        data[:kvals56.shape[0],8] = np.interp( kvals56, sd[4][::-1,0][maskNan], sd[4][::-1,1][maskNan] )
        data[kvals56.shape[0]:ptsPerPath[0],7:9] = np.nan
        # KMKp
        sd = self.sym_data[self.paths[1]]
        kk = self.kpoints_sym[self.paths[1]]
        kmin,kmax = (sd[0][-1,0],sd[3][-1,0])
        kvals = np.linspace(kmin,kmax,ptsPerPath[1])
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],0] = modK + modM - kvals
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],1] = np.interp( kvals, sd[0][:,0], kk[0][:,0] )
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],2] = np.interp( kvals, sd[0][:,0], kk[0][:,1] )
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],3] = np.interp( kvals, sd[0][:,0], sd[0][:,1] )
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],4] = np.interp( kvals, sd[1][:,0], sd[1][:,1] )
        data[ptsPerPath[0]:ptsPerPath[0]+ptsPerPath[1],5:9] = np.nan
        # KMKp close to M
        kmin,kmax = (sd[3][-1,0],sd[3][0,0])
        kvals = np.linspace(kmin,kmax,ptsPerPath[2])
        data[ptsPerPath[0]+ptsPerPath[1]:,0] = modK + modM - kvals
        data[ptsPerPath[0]+ptsPerPath[1]:,1] = np.interp( kvals, sd[0][:,0], kk[0][:,0] )
        data[ptsPerPath[0]+ptsPerPath[1]:,2] = np.interp( kvals, sd[0][:,0], kk[0][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,3] = np.interp( kvals, sd[0][:,0], sd[0][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,4] = np.interp( kvals, sd[1][:,0], sd[1][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,5] = np.interp( kvals, sd[2][:,0], sd[2][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,6] = np.interp( kvals, sd[3][:,0], sd[3][:,1] )
        data[ptsPerPath[0]+ptsPerPath[1]:,7:9] = np.nan

        # Order bands of index 1 and 2 which shifts at some point
        pp = ptsPerPath[0]+ptsPerPath[1]
        mask = data[pp:,4]<data[pp:,5]
        data[pp:,4][mask], data[pp:,5][mask] = data[pp:,5][mask], data[pp:,4][mask]
        # Shift of KMKp
        data[ptsPerPath[0]:,3] += self.offset[self.TMD]
        data[ptsPerPath[0]:,4] += self.offset[self.TMD]
        data[ptsPerPath[0]:,5] += self.offset[self.TMD]
        data[ptsPerPath[0]:,6] += self.offset[self.TMD]
        return data









