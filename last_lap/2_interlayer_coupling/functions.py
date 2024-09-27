import numpy as np
import sys,os
import numpy as np
import CORE_functions as cfs
from PIL import Image

#Moirè lattice length of bilayer, in Angstrom
a_Moire = 79.8

def energy(list_K,hopping,epsilon,HSO,offset,pars_interlayer,interlayer_type):
    en_list = np.zeros((list_K.shape[0],44))
    args_WSe2 = (hopping['WSe2'],epsilon['WSe2'],HSO['WSe2'],cfs.dic_params_a_mono['WSe2'],offset['WSe2'])
    args_WS2 = (hopping['WS2'],epsilon['WS2'],HSO['WS2'],cfs.dic_params_a_mono['WS2'],offset['WS2'])
    for k in range(list_K.shape[0]):
        big_H = np.zeros((44,44),dtype=complex)
        big_H[:22,:22] = cfs.H_monolayer(list_K[k],*args_WSe2)
        big_H[22:,22:] = cfs.H_monolayer(list_K[k],*args_WS2) 
        big_H += get_interlayer_H(list_K[k],pars_interlayer,interlayer_type)
        en_list[k] = np.linalg.eigvalsh(big_H)
    return en_list

def get_K(cut,n_pts):
    res = np.zeros((n_pts,2))
    a_mono = cfs.dic_params_a_mono['WSe2']
    if cut == 'KGK':
        K = np.array([4*np.pi/3,0])/a_mono
        for i in range(n_pts):
            res[i,0] = K[0]/(n_pts//2)*(i-n_pts//2)
    if cut == 'KMKp':
        M = np.array([np.pi,np.pi/np.sqrt(3)])/a_mono
        K = np.array([4*np.pi/3,0])/a_mono
        Kp = np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])/a_mono
        for i in range(n_pts//2):
            res[i] = K + (M-K)*i/(n_pts//2)
        for i in range(n_pts//2,n_pts):
            res[i] = M + (Kp-M)*i/(n_pts//2)
    return res

def get_interlayer_H(k,pars,interlayer_type):
    H = np.zeros((44,44),dtype=complex)
    if interlayer_type=='U1':
        t_k = -pars[0] + pars[1]*np.linalg.norm(k)**2
    elif interlayer_type=='C6':
        aa = cfs.dic_params_a_mono['WSe2']
        t_k = -pars[0] + pars[1]*2*(np.cos(k[0]*aa)+np.cos(k[0]/2*aa)*np.cos(np.sqrt(3)/2*k[1]*aa))
    elif interlayer_type=='C3':
        aa = cfs.dic_params_a_mono['WSe2']
        delta = aa*np.array([np.array([1,0]),np.array([1/2,np.sqrt(3)/2]),np.array([-1/2,np.sqrt(3)/2])])
        t_k = 0
        for i in range(3):
            t_k += pars[1]*np.exp(1j*np.dot(k,delta[i]))
    elif interlayer_type=='no':
        t_k = 0
    #a and b
    ind_pze = 8
    for i in range(2):
        H[ind_pze+11*i,ind_pze+11*i+22] = t_k 
    H += np.conjugate(H.T)
    #c
    for i in range(2):
        H[ind_pze+11*i+22,ind_pze+11*i+22] = pars[2]
    #Offset
    H += np.identity(44)*pars[-1]       
    return H

def extract_png(fig_fn,cut_bounds):
    pic_0 = np.array(np.asarray(Image.open(fig_fn)))
    #We go from -1 to 1 in image K cause the picture is stupid
    Ki = -1.4
    Kf = 1.4
    Ei = 0
    Ef = -3.5
    #Empirically extracted for S11 from -1 to +1
    P_ki = 810      #pixel corresponding to ki=-1
    P_kf = 2370     #pixel corresponding to ki=+1
    p_len = int((P_kf-P_ki)/2*(Kf-Ki))   #number of pixels from Ki to Kf
    p_ki = int((P_ki+P_kf)//2 - p_len//2)
    p_kf = int((P_ki+P_kf)//2 + p_len//2)
    #
    p_ei = 85       #pixel corresponding to ei=0
    p_ef = 1908     #pixel corresponding to ef=-3.5
    if len(cut_bounds) == 4:#Image cut
        ki,kf,ei,ef = cut_bounds
        pc_lenk = int(p_len/(Kf-Ki)*(kf-ki)) #number of pixels in cut image
        pc_ki = int((p_ki+p_kf)//2-pc_lenk//2)
        pc_kf = int((p_ki+p_kf)//2+pc_lenk//2)
        #
        pc_lene = int((p_ef-p_ei)/(Ei-Ef)*(ei-ef))
        pc_ei = p_ei + int((p_ef-p_ei)/(Ei-Ef)*(Ei-ei))
        pc_ef = p_ei + int((p_ef-p_ei)/(Ei-Ef)*(Ei-ef))
        return pic_0[pc_ei:pc_ef,pc_ki:pc_kf]
    else:
        return pic_0[p_ei:p_ef,p_ki:p_kf]

def plot_bands_on_exp(energies,pic,K_list,bounds,title=''):
    import matplotlib.pyplot as plt
    K,EM,Em = bounds
    plt.figure(figsize=(20,15))
    plt.imshow(pic)
    for i in range(22,28):
        plt.plot((K_list[:,0]+K)/2/K*pic.shape[1],(EM-energies[:,i])/(EM-Em)*pic.shape[0],'r-')
#    plt.xticks([0,pic.shape[1]//2,pic.shape[1]],["{:.2f}".format(-K),'0',"{:.2f}".format(K)])
    plt.xticks([0,pic.shape[1]//2,pic.shape[1]],[r"$K'$",r'$\Gamma$',r'$K$'],size=20)
    plt.yticks([0,pic.shape[0]//2,pic.shape[0]],["{:.2f}".format(EM),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(Em)])
#    plt.xlabel("$A^{-1}$",size=15)
    plt.ylabel("$E\;(eV)$",size=20)
    plt.title(title,size=20)
    plt.ylim(pic.shape[0],0)
#    plt.show()
    return plt.gcf()
#    plt.show()

def get_pars_fn(TMD,machine,dft=False):
    get_dft = '_DFT' if dft else '_fit'
    return get_home_dn(machine)+'inputs/pars_'+TMD+get_dft+'.npy'

def get_S11_fn(machine):
    return get_home_dn(machine)+'inputs/S11_KGK_WSe2onWS2_v1.png'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/last_lap/2_interlayer_coupling/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/2_interlayer_coupling/'
    elif machine == 'maf':
        pass

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
















