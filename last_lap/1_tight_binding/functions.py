import numpy as np
import CORE_functions as cfs
import parameters as ps
import scipy.linalg as la
from pathlib import Path
import os
import matplotlib.pyplot as plt
import itertools
#
def get_spec_args(ind):
    lP = np.linspace(0.05,0.15,6)
    lrp = np.linspace(1,3,3)
    lrl = np.linspace(0.1,0.5,3)
    lcv = [0]
    ll = [lP,lrp,lrl,lcv]
    combs = list(itertools.product(*ll))
    return combs[ind]

def chi2(pars,*args):
    """Compute square difference of bands with exp data.

    """
    data, TMD, machine, spec_args, ind = args
    H_SO = cfs.find_HSO(pars[-2:])
    tb_en = cfs.energy(pars,H_SO,data,TMD)
    #
    res = 0
    for b in range(2):
        args = np.argwhere(np.isfinite(data[b][:,1]))    #select only non-nan values
        res += np.sum(np.absolute(tb_en[b,args]-data[b][args,1])**2)
    par_dis = compute_parameter_distance(pars,np.array(ps.initial_pt[TMD]))
    final_res = res + spec_args[0]*par_dis
    #
    if final_res < ps.min_chi2 and ind>-1:   #remove old temp and add new one
        temp_fn = get_temp_fit_fn(TMD,ps.min_chi2,spec_args,ind,machine)
        if not ps.min_chi2==1e5:
            os.system('rm '+temp_fn)
        ps.min_chi2 = final_res
        temp_fn = get_temp_fit_fn(TMD,ps.min_chi2,spec_args,ind,machine)
        try:
            np.save(temp_fn,pars)
        except:
            print("Unable to write to file system, skipping this step")
    print(final_res)
    return final_res

def get_exp_data(TMD,machine):
    """For given material, takes the two cuts and the two bands and returns the lists of energy and momentum for the 2 top valence bands. 
    There are some NANs.

    """
    data = []
    offset_exp = {'WSe2':{'KGK':0,'KMKp':-0.0521}, 'WS2':{'KGK':0,'KMKp':-0.0025}} #To align the two cuts -> fixed on symmetrized data
    for cut in ['KGK','KMKp']:
        data.append([])
        for band in range(1,3):
            data_fn = get_exp_data_fn(TMD,cut,band,machine)
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
                    temp.append([float(ke[0]),float(ke[1])+offset_exp[TMD][cut],*find_vec_k(float(ke[0]),cut,TMD)])
            data[-1].append(np.array(temp))
            np.save(data_fn,np.array(temp))
    return data

def get_bounds(in_pt,spec_args):
    P, rp, rl, cv = spec_args
    Bounds = []
    off_ind = 3
    for i in range(len(in_pt)):     #tb parameters
        #
        if i == len(in_pt)-off_ind: #offset
            temp = (-3,0)
        elif i == len(in_pt)-2 or i == len(in_pt)-1: #SOC
            r = rl*in_pt[i]
            temp = (in_pt[i]-r,in_pt[i]+r)
        else:
            r = rp*abs(in_pt[i])
            mm, MM = (cv,0) if in_pt[i]>0 else (0,cv)
            temp = (in_pt[i]-r-mm,in_pt[i]+r+MM)
        Bounds.append(temp)
    return Bounds

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
            k_pts = M + (K-M)*np.abs(k_scalar)/la.norm(K-M)
        else:
            k_pts = M + (Kp-M)*np.abs(k_scalar)/la.norm(Kp-M)
    return k_pts

def get_spec_args_txt(spec_args):
    return "{:.3f}".format(spec_args[0]).replace('.',',')+'_'+"{:.3f}".format(spec_args[1]).replace('.',',')+'_'+"{:.3f}".format(spec_args[2]).replace('.',',')+'_'+"{:.3f}".format(spec_args[3]).replace('.',',')

def get_exp_data_fn(TMD,cut,band,machine):
    return get_exp_dn(machine)+'extracted_data_'+cut+'_'+TMD+'_band'+str(band)+'.npy'

def get_exp_fn(TMD,cut,band,machine):
    return get_exp_dn(machine)+cut+'_'+TMD+'_band'+str(band)+'.txt'

def get_temp_fit_fn(TMD,chi,spec_args,ind,machine):
    return get_temp_dn(machine,spec_args)+'pars_'+TMD+'_'+str(ind)+"_"+"{:.8f}".format(chi)+'.npy'

def get_res_fn(TMD,spec_args,machine):
    return get_res_dn(machine)+'res_'+TMD+'_'+get_spec_args_txt(spec_args)+'.npy'

def get_fig_fn(TMD,spec_args,machine):
    return get_fig_dn(machine)+'fig_'+TMD+'_'+get_spec_args_txt(spec_args)+'.png'

def get_fig_dn(machine):
    return get_res_dn(machine)+'figures/'

def get_exp_dn(machine):
    return get_home_dn(machine)+'inputs/'

def get_res_dn(machine):
    return get_home_dn(machine)+'results/'

def get_temp_dn(machine,spec_args):
    return get_res_dn(machine)+'temp_'+get_spec_args_txt(spec_args)+'/'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/1_tight_binding/'
    elif machine == 'maf':
        return '/users/rossid/1_tight_binding/'

def symmetrize(dataset):
    """datasìet has N k-entries, each containing a couple (k,E,kx,ky)"""
    new_ds = []
    len_ds = len(dataset)//2 if len(dataset)%2 == 0 else len(dataset)//2+1
    for i in range(len_ds):
        temp = np.zeros(4)
        temp[0] = np.abs(dataset[i,0])#np.sqrt(dataset[i,2]**2+dataset[i,3]**2)
        temp[2:] = dataset[i,2:]
        if np.isnan(dataset[i,1]) and np.isnan(dataset[-1-i,1]):
            temp[1] = np.nan
        elif np.isnan(dataset[i,1]):
            temp[1] = dataset[-1-i,1]
        elif np.isnan(dataset[-1-i,1]):
            temp[1] = dataset[i,1]
        else:
            temp[1] = (dataset[i,1]+dataset[-1-i,1])/2
        new_ds.append(temp)
    return np.array(new_ds)

def get_symm_data(exp_data):
    """ 2 bands, N k-points, (|k|,E,kx,ky)
    """
    symm_data = []
    for i in range(2):
        new_KGK = symmetrize(exp_data[0][i])
        new_KGK[:,2] *= -1
        new_KMK = symmetrize(exp_data[1][i])
        new_KMK[:,0] = exp_data[0][0][-1,0] - exp_data[1][0][0,0] - new_KMK[:,0]
        new = np.zeros((len(new_KGK)+len(new_KMK),4))
        new[:len(new_KGK)] = new_KGK[::-1]
        new[len(new_KGK):] = new_KMK
        symm_data.append(new)
    return symm_data

def get_reduced_data(symm_data,ind):
    red_data = []
    for i in range(2):
        red_data.append(symm_data[i][::ind])
    return red_data

def compute_parameter_distance(par,DFT):
    return np.sum(np.absolute(par[:-3]-DFT[:-3])**2) + np.sum(np.absolute(par[-2:]-DFT[-2:])**2)

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################

def ppe(a):
    return -1/np.sqrt(2)*(a[9]+1j*a[10])
def ppo(a):
    return -1/np.sqrt(2)*(a[3]+1j*a[4])
def pme(a):
    return  1/np.sqrt(2)*(a[9]-1j*a[10])
def pmo(a):
    return  1/np.sqrt(2)*(a[3]-1j*a[4])
def p0e(a):
    return a[8]
def p0o(a):
    return a[2]
def d0(a):
    return a[5]
def dp1(a):
    return -1/np.sqrt(2)*(a[0]+1j*a[1])
def dm1(a):
    return  1/np.sqrt(2)*(a[0]-1j*a[1])
def dp2(a):
    return  1/np.sqrt(2)*(a[7]+1j*a[6])
def dm2(a):
    return  1/np.sqrt(2)*(a[7]-1j*a[6])

fun = [ppe,ppo,pme,pmo,p0e,p0o,d0,dp1,dm1,dp2,dm2]
txt = ['ppe','ppo','pme','pmo','p0e','p0o','d0','dp1','dm1','dp2','dm2']

orb_txt = ['dxz','dyz','poz','pox','poy','dz2','dxy','dx2','pez','pex','pey']

def get_orbital_content(TMD,spec_args,machine):
    print("_____________________________________________________________________________")
    print("Orbital content:")
    a_mono = ps.dic_params_a_mono[TMD]
    k_pts = np.array([np.zeros(2),np.matmul(cfs.R_z(np.pi/3),np.array([4/3*np.pi/a_mono,0]))])    #Gamma and K (K+ of Fange et al., 2015)
    txt_pt = ['Gamma:','K:    ']
    fun_pt = [[d0,p0e],[dp2,ppe]]
    txt_fun_pt = [['d0 ','p0e'],['dp2','ppe']]
    #
    file = get_res_fn(TMD,spec_args,machine)
    full_pars = np.load(file)
    DFT_pars = np.array(ps.initial_pt[TMD])
    #
    args_DFT = (cfs.find_t(DFT_pars),cfs.find_e(DFT_pars),cfs.find_HSO(DFT_pars[-2:]),a_mono,DFT_pars[-3])
    H_DFT = cfs.H_monolayer(k_pts,*args_DFT)
    args_res = (cfs.find_t(full_pars),cfs.find_e(full_pars),cfs.find_HSO(full_pars[-2:]),a_mono,full_pars[-3])
    H_res = cfs.H_monolayer(k_pts,*args_res)
    #
    print("      \tDFT values\t\tres values")
    for i in range(2):
        H0 = H_DFT[i,11:,11:]    #Spinless Hamiltonian
        H1 = H_res[i,11:,11:]    #Spinless Hamiltonian
        k = k_pts[i]
        E0,evec0 = np.linalg.eigh(H0)
        E1,evec1 = np.linalg.eigh(H1)
        a0 = evec0[:,6]
        a1 = evec1[:,6]
        #
        v0 = [fun_pt[i][0](a0),fun_pt[i][1](a0)]
        v1 = [fun_pt[i][0](a1),fun_pt[i][1](a1)]
        print(txt_pt[i]+'\t'+txt_fun_pt[i][0]+':    '+"{:.3f}".format(np.absolute(v0[0]))+'\t\t'+txt_fun_pt[i][0]+':    '+"{:.3f}".format(np.absolute(v1[0])))
        print('      '+'\t'+txt_fun_pt[i][1]+':    '+"{:.3f}".format(np.absolute(v0[1]))+'\t\t'+txt_fun_pt[i][1]+':    '+"{:.3f}".format(np.absolute(v1[1])))
        print('      '+'\tweight: '+"{:.3f}".format(np.absolute(v0[0])**2+np.absolute(v0[1])**2)+'\t\tweight: '+"{:.3f}".format(np.absolute(v1[0])**2+np.absolute(v1[1])**2))
        if i == 0:
            print("________________________________________________________")




def get_table(TMD,spec_args,machine):
    print("_____________________________________________________________________________")
    print("Table of parameters with distance from DFT")
    file = get_res_fn(TMD,spec_args,machine)
    full_pars = np.load(file)
    pars_dft = ps.initial_pt[TMD]
    list_names = ps.list_names_all
    for i in range(len(pars_dft)):
        percentage = np.abs((full_pars[i]-pars_dft[i])/pars_dft[i]*100)
        l = 10 - len(list_names[i])
        sp1 = '' if percentage>10 else ' '
        sp2 = '' if pars_dft[i]<0 else ' '
        sp3 = '' if full_pars[i]<0 else ' '
        print(list_names[i],':',' '*l,sp2,"{:.5f}".format(pars_dft[i]),'  ->  ',sp3,"{:.5f}".format(full_pars[i]),'    ',sp1,"{:.2f}".format(percentage),'%')






