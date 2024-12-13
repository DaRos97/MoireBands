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

def get_interlayer_H(k,pars,interlayer_type):
    H = np.zeros((44,44),dtype=complex)
    if interlayer_type=='U1':
        t_k = -pars[0] + pars[1]*np.linalg.norm(k)**2
    elif interlayer_type=='C6':
        aa = cfs.dic_params_a_mono['WSe2']
        arr0 = aa*np.array([0,-1])
        t_k = -pars[0]
        for i in range(6):
            t_k += pars[1]*np.exp(1j*np.dot(k,np.dot(cfs.R_z(np.pi/3*i),arr0)))
#        t_k = -pars[0] + pars[1]*2*(np.cos(k[0]*aa)+np.cos(k[0]/2*aa)*np.cos(np.sqrt(3)/2*k[1]*aa))
    elif interlayer_type=='C3':
        aa = cfs.dic_params_a_mono['WSe2']
        ang = np.pi/3
        delta = aa*np.array([cfs.R_z(ang)@np.array([0,-1]),cfs.R_z(ang)@np.array([1/2,np.sqrt(3)/2]),cfs.R_z(ang)@np.array([-1/2,np.sqrt(3)/2])])
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

def extract_png(fig_fn,cut_bounds,sample):
    pic_0 = np.array(np.asarray(Image.open(fig_fn)))
    #We go from -1 to 1 in image K cause the picture is stupid
    Ki, Kf, Ei, Ef, P_ki, P_kf, p_ei, p_ef = cfs.dic_pars_samples[sample]
    #Empirically extracted for sample from -1 to +1
    p_len = int((P_kf-P_ki)/2*(Kf-Ki))   #number of pixels from Ki to Kf
    p_ki = int((P_ki+P_kf)//2 - p_len//2)
    p_kf = int((P_ki+P_kf)//2 + p_len//2)
    #
    if len(cut_bounds) == 4:#cut image
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

def get_SOC_fn(TMD,machine):
    return get_home_dn(machine)+'inputs/'+TMD+'_SOC.npy'

def get_pars_fn(TMD,machine,dft=False):
    get_dft = '_DFT' if dft else '_fit'
    return get_home_dn(machine)+'inputs/pars_'+TMD+get_dft+'.npy'

def get_sample_fn(sample,machine):
    v = 'v2' if sample == 'S3' else 'v1'
    return get_home_dn(machine)+'inputs/'+sample+'_KGK_WSe2onWS2_'+v+'.png'

def get_res_fn(title,int_type,machine):
    return get_home_dn(machine)+'results/'+title+'_'+int_type+'_pars_interlayer.npy'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/last_lap/2_interlayer_coupling/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/2_interlayer_coupling/'
    elif machine == 'maf':
        pass















