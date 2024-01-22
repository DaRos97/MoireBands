import numpy as np
import parameters as ps
import numpy.linalg as la
from pathlib import Path
import os

a_1 = np.array([1,0])
a_2 = np.array([-1/2,np.sqrt(3)/2])
J_plus = ((3,5), (6,8), (9,11))
J_minus = ((1,2), (3,4), (4,5), (6,7), (7,8), (9,10), (10,11))
J_MX_plus = ((3,1), (5,1), (4,2), (10,6), (9,7), (11,7), (10,8))
J_MX_minus = ((4,1), (3,2), (5,2), (9,6), (11,6), (10,7), (9,8), (11,8))

def chi2(pars,*args):
    """Compute square difference of bands with exp data.

    """
    exp_data, TMD, machine, range_par, cuts, plot = args
    tb_en = energy(pars,exp_data,cuts,TMD)
    res = 0
    for c in range(len(cuts)):
        for b in range(2):
            args = np.argwhere(np.isfinite(exp_data[c][b][:,1]))
            res += np.sum(np.absolute(tb_en[c][b,args]-exp_data[c][b][args,1])**2)
    if plot:
        plot_exp_tb(exp_data,tb_en,tb_en,TMD)
    if res < ps.min_chi2:
        os.system('rm '+get_temp_fit_fn(TMD,ps.min_chi2,range_par,cuts,machine))
        ps.min_chi2 = res
        np.save(get_temp_fit_fn(TMD,res,range_par,cuts,machine),pars)
    return res

def plot_exp_tb(exp_data,dft_en,tb_en,title=''):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(40,20))
    for b in range(2):
        for c in range(2):
            dft = (dft_en[c][b,:]-tb_en[c][b,:]).any()
            plt.subplot(2,2,2*b+c+1)
            plt.scatter(exp_data[c][b][:,0],exp_data[c][b][:,1],color='b',marker='*',label='experiment')
            plt.scatter(exp_data[c][b][:,0],tb_en[c][b,:],color='r',marker='.',label='minimization',s=5)
            if dft:
                plt.scatter(exp_data[c][b][:,0],dft_en[c][b,:],color='g',marker='^',label='DFT',s=5)
            plt.title('Cut '+ps.paths[c]+', band '+str(b))
            plt.legend()
    plt.suptitle(title)
    plt.show()

def plot_together(exp_data,dft_en,tb_en,title=''):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(40,20))
    for c in range(2):
        plt.subplot(1,2,c+1)
        for b in range(2):
            dft = (dft_en[c][b,:]-tb_en[c][b,:]).any()
            plt.scatter(exp_data[c][b][:,0],exp_data[c][b][:,1],color='b',marker='*',label='experiment')
            plt.scatter(exp_data[c][b][:,0],tb_en[c][b,:],color='r',marker='.',label='minimization',s=5)
            if dft:
                plt.scatter(exp_data[c][b][:,0],dft_en[c][b,:],color='g',marker='^',label='DFT',s=5)
        #plt.title('Cut '+ps.paths[c]+', band '+str(b))
        plt.legend()
    plt.suptitle(title)
    return plt.gcf()
#    plt.show()

def energy(parameters,data,cuts,TMD):
    """Compute energy along the two cuts of 2 TVB for all considered k.

    """
    hopping = find_t(parameters)
    epsilon = find_e(parameters)
    HSO = find_HSO(parameters)
    
    cut_energies = []
    offset = parameters[-1]
    for c in range(len(cuts)):
        kpts = data[c][0].shape[0]
        ens = np.zeros((2,kpts))
        for i in range(kpts):
            K = data[c][0][i,2:]
            H_mono = H_monolayer(K,hopping,epsilon,HSO,ps.dic_params_a_mono[TMD],offset)     #Compute UL Hamiltonian for given K
            temp = la.eigvalsh(H_mono)
            #index of TVB is 13, the other is 12 (out of 22: 11 bands times 2 for SOC. 7/11 are valence -> 14 is the TVB)
            ens[0,i] = temp[13]
            ens[1,i] = temp[12]
        cut_energies.append(ens)
    return cut_energies

def H_monolayer(K_p,hopping,epsilon,HSO,a_mono,offset):
    """Monolayer Hamiltonian.
    TO CHECK

    """
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

    #Offset
    H += np.identity(22)*offset
    return H

def get_exp_data(TMD,cuts,machine):
    """For given material, takes the two cuts and the two bands and returns the lists of energy and momentum for the 2 top valence bands. 
    There are some NANs.

    """
    data = []
    for cut in cuts:
        data.append([])
        for band in range(1,3):
            data_fn = get_ext_data_fn(TMD,cut,band,machine)
            if Path(data_fn).is_file():
                data[-1].append(np.load(data_fn))
                continue
            with open(get_exp_fn(TMD,cut,band,machine), 'r') as f:
                lines = f.readlines()
            temp = []
            for i in range(len(lines)):
                ke = lines[i].split('\t')
                if ke[1] == 'NAN\n':
                    temp.append([float(ke[0]),np.nan,*find_vec_k(float(ke[0]),cut,TMD)])
                else:
                    temp.append([float(ke[0]),float(ke[1]),*find_vec_k(float(ke[0]),cut,TMD)])
            data[-1].append(np.array(temp))
            np.save(data_fn,np.array(temp))
    return data

def find_vec_k(k_scalar,cut,TMD):
    """Compute vector k depending on modulus and cut.

    """
    a_mono = ps.dic_params_a_mono[TMD]
    k_pts = np.zeros(2)
    if cut == 'KGK':
        k_pts[0] = k_scalar
        k_pts[1] = 0
    elif cut == 'KMKp':
        M = np.array([np.pi,np.pi/np.sqrt(3)])/a_mono
        K = np.array([4*np.pi/3,0])/a_mono
        Kp = np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])/a_mono
        if k_scalar < 0:
            k_pts = M + (M-K)*np.abs(k_scalar)/la.norm(M-K)
        else:
            k_pts = M + (M-Kp)*np.abs(k_scalar)/la.norm(M-Kp)
    return k_pts

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

def find_HSO(dic_params_H):
    """Compute the SO Hamiltonian. TO CHECK.

    """
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

def get_ext_data_fn(TMD,cut,band,machine):
    return get_home_dn(machine)+'inputs/extracted_data_'+cut+'_'+TMD+'_band'+str(band)+'.npy'

def get_exp_fn(TMD,cut,band,machine):
    return get_home_dn(machine)+'inputs/'+cut+'_'+TMD+'_band'+str(band)+'_v1.txt'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/1_tight_binding/'
    elif machine == 'maf':
        pass

def get_fig_fn(TMD,cuts,range_par,machine):
    cuts_fn = ''
    for i in range(len(cuts)):
        cuts_fn += cuts[i]
        if i != len(cuts)-1:
            cuts_fn += '_'
    return get_home_dn(machine)+'results/fig_'+TMD+'_'+cuts_fn+'_'+"{:.2f}".format(range_par).replace('.',',')+'.png'

def get_fit_fn(range_par,TMD,res,cuts,machine):
    cuts_fn = ''
    for i in range(len(cuts)):
        cuts_fn += cuts[i]
        if i != len(cuts)-1:
            cuts_fn += '_'
    return get_home_dn(machine)+'results/pars_'+TMD+'_'+"{:.2f}".format(range_par).replace('.',',')+'_'+cuts_fn+'_'+"{:.4f}".format(res)+'.npy'

def get_temp_fit_fn(TMD,res,range_par,cuts,machine):
    cuts_fn = ''
    for i in range(len(cuts)):
        cuts_fn += cuts[i]
        if i != len(cuts)-1:
            cuts_fn += '_'
    return get_home_dn(machine)+'results/temp'+'/pars_'+TMD+'_'+"{:.2f}".format(range_par).replace('.',',')+'_'+cuts_fn+'_'+"{:.4f}".format(res)+'.npy'

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

def get_parameters(ind):
    TMDs = ['WSe2','WS2']
    cutss = [['KGK','KMKp'],['KGK',]]
    range_pars = np.linspace(0.1,1,10,endpoint=True)
    ind_tmd = ind//(len(cutss)*len(range_pars))
    ind_cut = ind%(len(cutss)*len(range_pars)) // len(range_pars)
    ind_rng = ind%(len(cutss)*len(range_pars)) % len(range_pars)
    return (TMDs[ind_tmd], cutss[ind_cut], range_pars[ind_rng])

def get_parameters_plot(ind):
    cutss = [['KGK','KMKp'],['KGK',]]
    range_pars = np.linspace(0.1,1,10,endpoint=True)
    ind_cut = ind // len(range_pars)
    ind_rng = ind % len(range_pars)
    return (cutss[ind_cut], range_pars[ind_rng])





















