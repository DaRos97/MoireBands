import numpy as np
import CORE_functions as cfs
import scipy.linalg as la
from pathlib import Path
import sys,os
import matplotlib.pyplot as plt
import itertools

"""temp value of chi 2"""
global min_chi2
global evaluation_step
min_chi2 = 1e5
evaluation_step = 0

def get_spec_args(ind):
    lP = [0.1,]
    lrp = [0.5]     #tb bounds
    lrl = [0,]      #SOC bounds
    lReduced = [13,]
    lPbc = [10,]
    lPdk = [20,]
    return list(itertools.product(*[cfs.TMDs,lP,lrp,lrl,lReduced,lPbc,lPdk]))[ind]

def chi2_SOC(pars_SOC,*args):
    """
    Compute square difference of bands with exp data.
    """
    reduced_data, other_pars, TMD, machine = args
    HSO = cfs.find_HSO(pars_SOC[1:])
    full_pars = np.append(other_pars,pars_SOC)
    tb_en = cfs.energy(full_pars,HSO,reduced_data,TMD)
    #
    result = 0
    k_pts = len(reduced_data[0])
    plot = False
    if plot:
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1,1,1)
    for b in range(2):
        farg = np.zeros(4,dtype=int)
        farg[0] = np.argmax(reduced_data[b][:k_pts//4,1])   #Max at Gamma
        farg[1] = np.argmin(reduced_data[b][:k_pts//2,1])   #Min between Gamma and K
        farg[2] = k_pts//2+np.argmax(reduced_data[b][k_pts//2:3*k_pts//4,1])    #max at K
        farg[3] = k_pts-2   #Min at M, -1 because M might be a bit weird
        #
        if plot:
            ax.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='*',label='new symm' if b == 0 else '')
            ax.plot(reduced_data[b][farg,0],tb_en[b,farg],color='g',marker='^',ls='-',label='fit' if b == 0 else '')
        #
        for i in [0,2]:  #just the maxima at gamma and K
            increase = np.absolute(tb_en[b,farg[i]]-reduced_data[b][farg[i],1])
            result += increase
            if plot:
                print(np.absolute(tb_en[b,farg[i]]-reduced_data[b][farg[i],1]))
                ax.scatter(reduced_data[b][farg[i],0],reduced_data[b][farg[i],1],c='k',marker='*',zorder=10,s=200)
    if plot:
        plt.show()
    return result

def chi2(pars_tb,*args):
    """Compute square difference of bands with exp data.

    """
    reduced_data, HSO, SOC_pars, machine, spec_args, ind_random = args
    full_pars = np.append(pars_tb,SOC_pars[-2:])
    #Compute energy of new pars
    tb_en = cfs.energy(full_pars,HSO,reduced_data,spec_args[0])
    #
    result = 0
    #chi2 of bands distance
    for b in range(2):
        result += np.sum(np.absolute(tb_en[b]-reduced_data[b][:,1])**2)
    #chi2 of parameters distance
    par_dis = compute_parameter_distance(pars_tb,spec_args[0])
    result += spec_args[1]*par_dis
    #chi2 of bands' content
    band_content = np.array(compute_band_content(full_pars,HSO,spec_args[0]))
    Pbc = spec_args[5]
    result += Pbc*(2-np.sum(np.absolute(band_content)**2))
    #chi2 of distance at Gamma and K
    Pdk = spec_args[6]
    indexes = [0,27]    #indexes of Gamma and K for ind_reduced=14
    for i in range(2):  #2 bands
        for j in range(2):  #Gamma and K
            result += Pdk*(np.absolute(tb_en[i,indexes[j]]-reduced_data[i][indexes[j],1])**2)
    #Save temporary file if result goes down
    global min_chi2
    if result < min_chi2 and ind_random >= 0:   #remove old temp and add new one
        temp_fn = get_temp_fit_fn(min_chi2,spec_args,ind_random,machine)
        if not min_chi2==1e5:
            os.system('rm '+temp_fn)
        min_chi2 = result
        temp_fn = get_temp_fit_fn(min_chi2,spec_args,ind_random,machine)
        np.save(temp_fn,pars_tb)
    #Plot figure every N steps to see how it is going
    if machine=='loc':    #Plot each nnnn steps
        global evaluation_step
        evaluation_step += 1
        nnnn = 1000
        if evaluation_step%nnnn==0:
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(1,1,1)
            for b in range(2):
                ax.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='*',label='new symm' if b == 0 else '')
                targ = np.argwhere(np.isfinite(reduced_data[b][:,1]))    #select only non-nan values
                ax.plot(reduced_data[b][targ,0],tb_en[b,targ],color='g',marker='^',ls='-',label='fit' if b == 0 else '')
            #
            ax.set_xlabel(r'$A^{-1}$')
            ax.set_ylabel('E(eV)')
            ax.set_title("chi2: "+"{:.4f}".format(result))
            plt.legend()
            plt.savefig('results/figures/temp.png')
            plt.close(fig)
            print("New fig ",evaluation_step//nnnn,", chi2: ","{:.8f}".format(result))

            #fig of distance from DFT values
            fig = plt.figure(figsize=(15,20))
            ax1 = fig.add_subplot(2,1,1)
            ax1.bar(np.arange(len(pars_tb)),pars_tb-cfs.initial_pt[spec_args[0]][:-2],color='r')
            ax1.set_ylabel("Absolute")
            ax2 = fig.add_subplot(2,1,2)
            ax2.bar(np.arange(len(pars_tb)),(pars_tb-cfs.initial_pt[spec_args[0]][:-2])/abs(np.array(cfs.initial_pt[spec_args[0]][:-2]))*100,color='b')
            ax2.set_ylabel("Percentage")
            ax.set_title("chi2: "+"{:.4f}".format(result))
            fig.tight_layout()
            plt.savefig('results/figures/memp.png')
            plt.close(fig)
    #print("chi2: ","{:.7f}".format(result))
    return result

def compute_band_content(parameters,HSO,TMD):
    """
    Computes the band content of d0 and p0e at Gamma and the band content of dp2 and ppe at K.
    Returns a list with the coefficients: (c1t,c1,c6t,c6) in the notation of Fang et al., which
    are the components of d0,p0e,dp2,ppe respectvely.
    """
    functions_kpt = [[d0,p0e],[dp2,ppe]]
    args_H = (cfs.find_t(parameters),cfs.find_e(parameters),HSO,cfs.dic_params_a_mono[TMD],parameters[-3])
    k_pts = np.array([np.zeros(2),cfs.R_z(np.pi/3)@np.array([4/3*np.pi/cfs.dic_params_a_mono[TMD],0])])
    H = cfs.H_monolayer(k_pts,*args_H)
    result = []
    for i in range(k_pts.shape[0]): #kpt
        evals,evecs = np.linalg.eigh(H[i,:11,:11])
        for j in range(2):  #2 orbitals at each k_pt
            result.append(functions_kpt[i][j](evecs[:,6]))
    return result

def get_exp_data(TMD,machine):
    """
    For given material, takes the two cuts (KGK and KMKp) and the two bands and returns the lists of energy and momentum for the 2 top valence bands.
    We take from the .txt experiment data which has value of |k| and energy and save it as a .npy matrix with values:
        |k|, energy, kx, ky.
    Need to handle some NANs in the energy -> that point is not available -> still keep it.
    """
    data = []
    offset_exp = {
        'WSe2':{'KGK':0,'KMKp':-0.0521},
        'WS2':{'KGK':0,'KMKp':-0.0025}
    } #To align the two cuts -> fixed on symmetrized data
    for cut in ['KGK','KMKp']:
        data.append([])
        for band in range(1,3):
            data_fn = get_exp_data_fn(TMD,cut,band,machine)     #.npy file saved
            if Path(data_fn).is_file():
                data[-1].append(np.load(data_fn))
                continue
            with open(get_exp_fn(TMD,cut,band,machine), 'r') as f:  #original .txt file
                lines = f.readlines()
            temp = []
            for i in range(len(lines)):
                ke = lines[i].split('\t')       #momentum modulus and energy
                if ke[1] == 'NAN\n':
                    temp.append([float(ke[0]),np.nan,*find_vec_k(float(ke[0]),cut,TMD)])
                else:
                    temp.append([float(ke[0]),float(ke[1])+offset_exp[TMD][cut],*find_vec_k(float(ke[0]),cut,TMD)])
            data[-1].append(np.array(temp))
            np.save(data_fn,np.array(temp))
    return data

def get_symm_data(exp_data):
    """
    Symmetrize experimental data from k to -k, for the 2 cuts and the 2 bands.
    Experimental values of |k| are symmetric around 0, so each point has a symmetric one.
    We average between the two. If one of them is nan keep only the other. If both are nan give nan.
    We also put the data on a line G-K-M so the result is a 2xN matrix for the 2 bands.
    """
    Nkgk = len(exp_data[0][0])
    Nkmk = len(exp_data[1][0])
    symm_data = np.zeros((2,Nkgk//2+Nkmk//2+Nkmk%2,4))
    for i in range(2):  #two bands
        for ik in range(Nkgk//2,Nkgk):       #second half for kgk
            if np.isnan(exp_data[0][i][ik][1]):
                symm_data[i,ik-Nkgk//2] = exp_data[0][i][Nkgk-1-ik]
            else:
                if np.isnan(exp_data[0][i][Nkgk-1-ik][1]):
                    symm_data[i][ik-Nkgk//2] = exp_data[0][i][ik]
                else:       #actuall average
                    symm_data[i,ik-Nkgk//2] = np.array([exp_data[0][i][ik][0],(exp_data[0][i][ik][1]+exp_data[0][i][Nkgk-1-ik][1])/2,exp_data[0][i][ik][2],exp_data[0][i][ik][3]])
        for ik in range(Nkmk//2+Nkmk%2):       #first half for kmk
            if np.isnan(exp_data[1][i][ik][1]):
                symm_data[i,Nkgk//2+ik] = exp_data[1][i][Nkmk-1-ik]
            else:
                if np.isnan(exp_data[1][i][Nkmk-1-ik][1]):
                    symm_data[i][ik+Nkgk//2] = exp_data[1][i][ik]
                else:       #actuall average
                    symm_data[i,ik+Nkgk//2] = np.array([exp_data[1][i][ik][0],(exp_data[1][i][ik][1]+exp_data[1][i][Nkmk-1-ik][1])/2,exp_data[1][i][ik][2],exp_data[1][i][ik][3]])
        symm_data[i,Nkgk//2:,0] += symm_data[i,Nkgk//2-1,0] + exp_data[1][i][-1][0]
    return symm_data

def get_reduced_data(symm_data,ind):
    """
    Get reduced set of k-points for the comparison.
    """
    red_data = []
    for i in range(2):
        red_data.append(np.concatenate((symm_data[i][::ind],[symm_data[i][-1],]),axis=0))
    return red_data

def symmetrize(dataset):
    """dataset has N k-entries, each containing a couple (k,E,kx,ky)"""
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
#            temp[1] = (dataset[i,1]+dataset[-1-i,1])/2
            temp[1] = dataset[i,1]
        new_ds.append(temp)
    return np.array(new_ds)

def get_bounds(in_pt,spec_args):
    TMD, P, rp, rl, ind_reduced, Pbc, Pdk = spec_args
    Bounds = []
    for i in range(in_pt.shape[0]):     #tb parameters
        if i == in_pt.shape[0]-3: #offset
            temp = (-3,0)
        elif i == in_pt.shape[0]-2 or i == in_pt.shape[0]-1: #SOC
            r = rl*in_pt[i]
            temp = (in_pt[i]-r,in_pt[i]+r)
        else:
            r = rp*abs(in_pt[i])
            temp = (in_pt[i]-r,in_pt[i]+r)
        Bounds.append(temp)
    return Bounds

def find_vec_k(k_scalar,cut,TMD):
    """
    Compute vector components from the (signed)modulus depending cut and TMD.
    """
    a_mono = cfs.dic_params_a_mono[TMD]
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
    return spec_args[0]+'_'+"{:.3f}".format(spec_args[1]).replace('.',',')+'_'+"{:.3f}".format(spec_args[2]).replace('.',',')+'_'+"{:.3f}".format(spec_args[3]).replace('.',',')+'_'+str(spec_args[4])+'_'+str(spec_args[5])+'_'+str(spec_args[6])

def get_exp_data_fn(TMD,cut,band,machine):
    return get_exp_dn(machine)+'extracted_data_'+cut+'_'+TMD+'_band'+str(band)+'.npy'

def get_exp_fn(TMD,cut,band,machine):
    return get_exp_dn(machine)+cut+'_'+TMD+'_band'+str(band)+'.txt'

def get_temp_fit_fn(chi,spec_args,ind_random,machine):
    return get_temp_dn(machine,spec_args)+'temp_'+str(ind_random)+"_"+"{:.8f}".format(chi)+'.npy'

def get_res_fn(spec_args,machine):
    return get_res_dn(machine)+'result_'+get_spec_args_txt(spec_args)+'.npy'

def get_fig_fn(spec_args,machine):
    return get_fig_dn(machine)+'fig_'+get_spec_args_txt(spec_args)+'.png'

def get_SOC_fn(TMD,machine):
    return get_res_dn(machine)+TMD+'_SOC.npy'

def get_fig_dn(machine):
    return get_res_dn(machine)+'figures/'

def get_exp_dn(machine):
    return get_home_dn(machine)+'inputs/'

def get_res_dn(machine):
    return get_home_dn(machine)+'Data/'

def get_temp_dn(machine,spec_args):
    return get_res_dn(machine)+'temp_'+get_spec_args_txt(spec_args)+'/'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/Code/1_monolayer/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/1_monolayer/'
    elif machine == 'maf':
        return '/users/rossid/1_monolayer/'

def compute_parameter_distance(pars,TMD):
    DFT_values = np.array(cfs.initial_pt[TMD])
    len_tb = DFT_values.shape[0]
    if pars.shape[0]==len_tb:
        return np.sum(np.absolute(pars[:-3]-DFT_values[:-3])**2) + np.sum(np.absolute(pars[-2:]-DFT[-2:])**2)
    elif pars.shape[0]==len_tb-2:
        return np.sum(np.absolute(pars-DFT_values[:-2])**2)
    else:
        print("compute_parameter_distance error")

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

orb_txt = ['dxz','dyz','poz','pox','poy','dz2','dxy','dx2','pez','pex','pey']

def get_orbital_content(spec_args,machine,fn=''):
    print("_____________________________________________________________________________")
    print("Orbital content:")
    a_mono = cfs.dic_params_a_mono[spec_args[0]]
    k_pts = np.array([np.zeros(2),np.matmul(cfs.R_z(np.pi/3),np.array([4/3*np.pi/a_mono,0]))])    #Gamma and K (K+ of Fange et al., 2015)
    txt_pt = ['Gamma:','K:    ']
    fun_pt = [[d0,p0e],[dp2,ppe]]
    txt_fun_pt = [['d0 ','p0e'],['dp2','ppe']]
    #
    if fn=='':
        fn = get_res_fn(spec_args,machine)
    full_pars = np.load(fn)
    DFT_pars = np.array(cfs.initial_pt[spec_args[0]])
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

def get_table(spec_args,machine,fn=''):
    print("_____________________________________________________________________________")
    print("Table of parameters with distance from DFT")
    if fn=='':
        fn = get_res_fn(spec_args,machine)
    full_pars = np.load(fn)
    pars_dft = cfs.initial_pt[spec_args[0]]
    list_names = cfs.list_names_all
    for i in range(len(full_pars)):
        percentage = np.abs((full_pars[i]-pars_dft[i])/pars_dft[i]*100)
        l = 10 - len(list_names[i])
        sp1 = '' if percentage>10 else ' '
        sp2 = '' if pars_dft[i]<0 else ' '
        sp3 = '' if full_pars[i]<0 else ' '
        print(list_names[i],':',' '*l,sp2,"{:.5f}".format(pars_dft[i]),'  ->  ',sp3,"{:.5f}".format(full_pars[i]),'    ',sp1,"{:.2f}".format(percentage),'%')






