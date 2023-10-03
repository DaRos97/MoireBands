import numpy as np
ind_res = 0
list_res_bm = [1/(2**x) for x in range(30)]
#####
#####Parameters
#####
###Monolayer paerameters
#WS2 --> Table III  (first two in Angstrom, all others in eV, last in .. (lambda of SO)
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

            #SO
            #'W':
            0.2874,
            #'S'
            0.0556,

            #'offset
            0.31,

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

            #SO
            #'W':
            0.2874,
            #'Se'
            0.2470,

            #'offset
            0.46,
                ],
        }

def find_t(dic_params_H):
    #Define hopping matrix elements from inputs and complete all symmetry related ones
    t = []
    t.append(np.zeros((11,11))) #t1
    t.append(np.zeros((11,11))) #t2
    t.append(np.zeros((11,11))) #t3
    t.append(np.zeros((11,11))) #t4
    t.append(np.zeros((11,11))) #t5
    t.append(np.zeros((11,11))) #t6
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
    #Now add non-independent parameters
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
    #Define the array of on-site energies from inputs and symmetry related ones
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

#Spin-Orbit Hamiltonian
#It is k-independent so depends only on \lambda of the materials in the TMD -> 2 parameters for each layer
dic_params_SO = {
        'W': 0.2874,
        'S': 0.0556,
        'Se': 0.2470,
        }
def find_HSO(dic_params_H):
    l_M = dic_params_H[40]
    l_X = dic_params_H[41]
    ####
    Mee_uu = np.zeros((6,6),dtype=complex)
    Mee_uu[1,2] = 1j*l_M
    Mee_uu[2,1] = -1j*l_M
    Mee_uu[4,5] = -1j*l_X/2
    Mee_uu[5,4] = 1j*l_X/2
    Mee_dd = -Mee_uu
    #
    Moo_uu = np.zeros((5,5),dtype=complex)
    Moo_uu[0,1] = -1j*l_M/2
    Moo_uu[1,0] = 1j*l_M/2
    Moo_uu[3,4] = -1j*l_X/2
    Moo_uu[4,3] = 1j*l_X/2
    Moo_dd = -Moo_uu
    #
    Moe_ud = np.zeros((5,6),dtype=complex)
    Moe_ud[0,0] = l_M*np.sqrt(3)/2
    Moe_ud[0,1] = 1j*l_M/2
    Moe_ud[0,2] = -l_M/2
    Moe_ud[1,0] = -1j*l_M*np.sqrt(3)/2
    Moe_ud[1,1] = -l_M/2
    Moe_ud[1,2] = -1j*l_M/2
    Moe_ud[2,4] = -l_X/2
    Moe_ud[2,5] = 1j*l_X/2
    Moe_ud[3,3] = l_X/2
    Moe_ud[4,3] = -1j*l_X/2
    Meo_du = np.conjugate(Moe_ud.T)
    #
    Meo_ud = np.zeros((6,5),dtype=complex)
    Meo_ud[0,0] = -l_M*np.sqrt(3)/2
    Meo_ud[0,1] = 1j*l_M*np.sqrt(3)/2
    Meo_ud[1,0] = -1j*l_M/2
    Meo_ud[1,1] = l_M/2
    Meo_ud[2,0] = l_M/2
    Meo_ud[2,1] = 1j*l_M/2
    Meo_ud[3,3] = -l_X/2
    Meo_ud[3,4] = 1j*l_X/2
    Meo_ud[4,2] = l_X/2
    Meo_ud[5,2] = -1j*l_X/2
    Moe_du = np.conjugate(Meo_ud.T)
    #
    Muu = np.zeros((11,11),dtype=complex)
    Muu[:5,:5] = Moo_uu
    Muu[5:,5:] = Mee_uu
    Mdd = np.zeros((11,11),dtype=complex)
    Mdd[:5,:5] = Moo_dd
    Mdd[5:,5:] = Mee_dd
    Mud = np.zeros((11,11),dtype=complex)
    Mud[:5,5:] = Moe_ud
    Mud[5:,:5] = Meo_ud
    Mdu = np.zeros((11,11),dtype=complex)
    Mdu[:5,5:] = Moe_du
    Mdu[5:,:5] = Meo_du
    #
    HSO = np.zeros((22,22),dtype=complex)
    HSO[:11,:11] = Muu
    HSO[11:,11:] = Mdd
    HSO[:11,11:] = Mud
    HSO[11:,:11] = Mdu
    ####
    return HSO
#Angstrom
dic_params_a_mono = {
        'WS2': 3.18,
        'WSe2': 3.32,
        }
###Moirè potentials of bilayers. Different at Gamma (d_z^2 orbital) -> first two parameters, and K (d_xy orbitals) -> last two parameters
#WS2/WSe2 --> Gamma points from paper "G valley TMD moirè bands"(first in eV, second in radiants)
#WS2/WSe2 --> Louk's paper for K points(first in eV, second in radiants)
dic_params_V = {'WSe2/WS2':[0.0335,np.pi, 7.7*1e-3, -106*2*np.pi/360],
            }
###Moirè length of bilayers in Angstrom
dic_a_Moire = { 'WS2/WSe2':79.8,
            'WSe2/WS2':79.8,
       }

