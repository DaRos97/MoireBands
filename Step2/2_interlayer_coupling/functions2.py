import numpy as np
import sys,os
import numpy as np
import CORE_functions as cfs
from PIL import Image

def energy(list_K,hopping,epsilon,HSO,offset,pars_interlayer,interlayer_type):
    en_list = np.zeros((list_K.shape[0],44))
    args_WSe2 = (hopping['WSe2'],epsilon['WSe2'],HSO['WSe2'],cfs.dic_params_a_mono['WSe2'],offset['WSe2'])
    args_WS2 = (hopping['WS2'],epsilon['WS2'],HSO['WS2'],cfs.dic_params_a_mono['WS2'],offset['WS2'])
    for k in range(list_K.shape[0]):
        big_H = np.zeros((44,44),dtype=complex)
        big_H[:22,:22] = cfs.H_monolayer(list_K[k],*args_WSe2)
        big_H[22:,22:] = cfs.H_monolayer(list_K[k],*args_WS2)
        H_int = get_interlayer_H(list_K[k],pars_interlayer,interlayer_type)
        if np.max(np.abs(H_int-H_int.T.conj()))>1e-5:
            print("errrrrr ",interlayer_type)
        big_H += H_int
        en_list[k] = np.linalg.eigvalsh(big_H)
    return en_list

def get_interlayer_H(k,pars_interlayer,interlayer_type):
    H = np.zeros((44,44),dtype=complex)
    if interlayer_type=='U1':
        t_k = -pars_interlayer[0] + pars_interlayer[1]*np.linalg.norm(k)**2
    elif interlayer_type=='C6':
        aa = cfs.dic_params_a_mono['WSe2']
        arr0 = aa*np.array([1,0])
        t_k = -pars_interlayer[0]
        for i in range(6):
            t_k += pars_interlayer[1]*np.exp(1j*k@cfs.R_z(np.pi/3*i)@arr0)
    elif interlayer_type=='C3':
        aa = cfs.dic_params_a_mono['WSe2']
        arr0 = aa*np.array([1,0])/np.sqrt(3)
        t_k = 0
        for i in range(3):
            t_k += pars_interlayer[1]*np.exp(1j*k@cfs.R_z(2*np.pi/3*i)@arr0)
    elif interlayer_type=='no':
        t_k = 0
    #a and b
    ind_pze = 8
    for i in range(2):
        H[ind_pze+11*i,ind_pze+11*i+22] = t_k
    H += np.conjugate(H.T)
    #c
    for i in range(2):
        H[ind_pze+11*i+22,ind_pze+11*i+22] = pars_interlayer[2]
    #Offset
    H += np.identity(44)*pars_interlayer[3]
    return H

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

def get_SOC_fn(TMD,machine):
    return get_home_dn(machine)+'inputs/'+TMD+'_SOC.npy'

def get_pars_fn(TMD,machine,monolayer_type):
    return get_home_dn(machine)+'inputs/pars_'+TMD+'_'+monolayer_type+'.npy'

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















