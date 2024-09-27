import numpy as np
import scipy.linalg as la

"""Functions related to Monolayer Hamiltonian"""

a_1 = np.array([1,0])
a_2 = np.array([-1/2,np.sqrt(3)/2])
J_plus = ((3,5), (6,8), (9,11))
J_minus = ((1,2), (3,4), (4,5), (6,7), (7,8), (9,10), (10,11))
J_MX_plus = ((3,1), (5,1), (4,2), (10,6), (9,7), (11,7), (10,8))
J_MX_minus = ((4,1), (3,2), (5,2), (9,6), (11,6), (10,7), (9,8), (11,8))

TMDs = ['WSe2','WS2']

def energy(parameters,HSO,data,TMD):
    """Compute energy along the two cuts of 2 TopValenceBand for all considered k.

    """
    hopping = find_t(parameters)
    epsilon = find_e(parameters)
    a_TMD = dic_params_a_mono[TMD]
    offset = parameters[-3]
    #
    args_H = (hopping,epsilon,HSO,a_TMD,offset)
    #
    kpts = data[0].shape[0]
    all_H = H_monolayer(np.array(data[0][:,2:]),*args_H)
    ens = np.zeros((2,kpts))
    for i in range(kpts):
        #index of TVB is 13, the other is 12 (out of 22: 11 bands times 2 for SOC. 7/11 are valence -> 14 is the TVB)
        ens[:,i] = la.eigvalsh(all_H[i])[12:14][::-1]
#        Ens = Parallel(n_jobs=16)(delayed(egvals)(H_i) for H_i in all_H)
    return ens

def H_monolayer(K_p,*args):
    """Monolayer Hamiltonian.       TO CHECK

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
    Mee_uu[3,4] = -1j*l_X/2
    Mee_uu[4,3] = 1j*l_X/2
    Mee_dd = -Mee_uu
    #
    Moo_uu = np.zeros((5,5),dtype=complex)
    Moo_uu[0,1] = -1j*l_M/2
    Moo_uu[1,0] = 1j*l_M/2
    Moo_uu[2,3] = -1j*l_X/2
    Moo_uu[3,2] = 1j*l_X/2
    Moo_dd = -Moo_uu
    #
    Meo_ud = np.zeros((6,5),dtype=complex)
    Meo_ud[0,0] = -l_M*np.sqrt(3)/2
    Meo_ud[0,1] = 1j*l_M*np.sqrt(3)/2
    Meo_ud[1,0] = l_M/2
    Meo_ud[1,1] = 1j*l_M/2
    Meo_ud[2,0] = -1j*l_M/2
    Meo_ud[2,1] = l_M/2
    Meo_ud[3,4] = l_X/2
    Meo_ud[4,4] = -1j*l_X/2
    Meo_ud[5,2] = -l_X/2
    Meo_ud[5,3] = 1j*l_X/2
    Moe_du = np.conjugate(Meo_ud.T)
    #
    Meo_du = np.zeros((6,5),dtype=complex)
    Meo_du[0,0] = l_M*np.sqrt(3)/2
    Meo_du[0,1] = 1j*np.sqrt(3)*l_M/2
    Meo_du[1,0] = -l_M/2
    Meo_du[1,1] = -1j*l_M/2
    Meo_du[2,0] = -1j*l_M/2
    Meo_du[2,1] = -l_M/2
    Meo_du[3,4] = -l_X/2
    Meo_du[4,4] = -1j*l_X/2
    Meo_du[5,2] = l_X/2
    Meo_du[5,3] = 1j*l_X/2
    Moe_ud = np.conjugate(Meo_du.T)
    #
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
    P = np.zeros((11,11))
    P[0,5] = P[1,7] = P[2,6] = P[3,9] = P[4,10] = P[5,8] = 1
    P[6,0] = P[7,1] = P[8,3] = P[9,4] = P[10,2] = 1
    Pf = np.zeros((22,22))
    Pf[:11,:11] = P
    Pf[11:,11:] = P
    HSOf = np.linalg.inv(Pf) @ HSO @ Pf
    return HSOf

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

"""Utility functions"""

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

def tqdm(x):
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
    return 1/np.sqrt(1/dic_params_a_mono['WSe2']**2+1/dic_params_a_mono['WS2']**2-2*np.cos(theta)/dic_params_a_mono['WSe2']/dic_params_a_mono['WS2'])

def miniBZ_rotation(theta):
    return np.arctan(-np.tan(theta/2)*(dic_params_a_mono['WSe2']+dic_params_a_mono['WS2'])/(dic_params_a_mono['WSe2']-dic_params_a_mono['WS2']))

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
            -1.34,

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
            -0.73,

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

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

"""Parameters of experimental image

In order, are: 
    K_initial, K_final, E_initial, E_final,
    pixel of k=-1, pixel of k=1, pixel of e=E_i, pixel of e=E_F

"""

dic_pars_samples = {
        'S11': [-1.4, 1.4, 0, -3.5,
                810, 2371, 89, 1899],
        'S3':  [-1.4, 1.4, 0, -2.5,
                697, 2156, 108, 1681],
        'S11zoom': [-1.4, 1.4, -0.6, -1.8,
                840, 2980, 86, 1147],
        }



