"""Description of the script:
py tight_binding.py arg1 arg2
arg1: index of specification arguments, which includes
    - M -> material
    - P -> parameter of chi2_1
    - rp -> bounds of energies and hoppings
    - rl -> bounds of SOC
arg2: index of random realization within '5%' of DFT values

Description of the code:
We extract the experimental data, adjust it with an offset and symmetrize it, finally we take 1 every ind_reduced points to speed up the computation.
We extract the DFT parameters and compute initial point and the bounds.
Compute the chi squared function by evaluating the bands an minimizing it within the bounds.
"""
import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions1 as fs
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time import time as ttt
from datetime import timedelta

disp = True
plot_exp = False
fit_SOC = False
plot_SOC_fit = True if fit_SOC else False
save_SOC = True
plot_final_fit = False

t_initial = ttt()
machine = cfs.get_machine(os.getcwd())

ind_spec_args = 0 if len(sys.argv)==1 else int(sys.argv[1])
ind_random = 0 if len(sys.argv)<3 else int(sys.argv[2])

spec_args = fs.get_spec_args(ind_spec_args)
TMD = spec_args[0]
ind_reduced = spec_args[-1]

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,machine)
symm_data = fs.get_symm_data(exp_data)
reduced_data = fs.get_reduced_data(symm_data,ind_reduced)
if disp:
    print("------------CHOSEN PARAMETERS------------")
    print(" TMD: ",spec_args[0],"\n chi2_1 parameter: ","{:.4f}".format(spec_args[1]),"\n Bound parameters: ","{:.2f}".format(spec_args[2]*100)+"%","\n Bounds SOC: ","{:.2f}".format(spec_args[3]*100)+"%","\n Index random evaluation: ",ind_random)
    print(" Using 1 every ",ind_reduced," points, for a total of ",len(reduced_data[0])," points")
if plot_exp: #plot experimental data to see if they are aligned
    plt.figure(figsize=(14,7))
    plt.title(TMD)
    KGK_end = exp_data[0][0][-1,0]
    KMKp_beg = exp_data[1][0][0,0]
    ikl = exp_data[0][0].shape[0]//2+1
    for b in range(2):
        plt.plot(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
        plt.plot(exp_data[1][b][:,0]+KGK_end-KMKp_beg,exp_data[1][b][:,1],color='b',marker='*')
        #
        plt.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='*',label='symmetrized' if b == 0 else '')
    #
    plt.xlabel(r'$A^{-1}$')
    plt.ylabel('E(eV)')
    plt.legend()
    plt.show()
    exit()

#DFT values of tb parameters
DFT_values = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC

SOC_fn = fs.get_SOC_fn(TMD,machine)
if fit_SOC:
    print("Fitting SOC and offset at Gamma and K.")
    """
    We start by computing offset and SOC parameters by fitting the energy of the 2 top bands at Gamma and K.
    """
    args_chi2_SOC = (reduced_data, DFT_values[:-3], spec_args[0], machine)
    if not Path(SOC_fn).is_file():
        print("Computing fit of SOC")
        initial_point_SOC = DFT_values[-3:]
        lb = -1  #lower bound (%)
        ub = 2  #upper bound (%)
        Bounds_SOC = ((DFT_values[-3]*ub,DFT_values[-3]*lb),
                      (DFT_values[-2]*lb,DFT_values[-2]*ub),
                      (DFT_values[-1]*lb,DFT_values[-1]*ub))
        result_SOC = minimize(fs.chi2_SOC,
                args = args_chi2_SOC,
                x0 = initial_point_SOC,
                bounds = Bounds_SOC,
                method = 'Nelder-Mead',
                options = {
                    'disp': False,
                    'adaptive' : False,
                    'fatol': 1e-6,
                    'xatol': 1e-8,
                    'maxiter': 1e6,
                    },
                )
        SOC_pars = result_SOC.x
        if save_SOC:
            np.save(SOC_fn,SOC_pars[-3:])
    else:
        SOC_pars = np.load(SOC_fn)
    #
    if plot_SOC_fit:    #Plot result
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1,1,1)
        print("Computed SOC (offset, L_WSe2, L_WS2):")
        print(DFT_values[-3:])
        print("-->")
        print(SOC_pars)
        print("Chi^2 distance: ",fs.chi2_SOC(SOC_pars,*args_chi2_SOC))
    #    ax.set_title("{:.7f}".format(result_SOC.fun))
        KGK_end = exp_data[0][0][-1,0]
        KMKp_beg = exp_data[1][0][0,0]
        ikl = exp_data[0][0].shape[0]//2+1
        HSO = cfs.find_HSO(SOC_pars[1:])
        full_pars = np.append(DFT_values[:-3],SOC_pars)
        tb_en = cfs.energy(full_pars,HSO,reduced_data,spec_args[0])
        tb_en2 = cfs.energy(DFT_values,cfs.find_HSO(DFT_values[-2:]),reduced_data,spec_args[0])
        for b in range(2):
            if 0:
                ax.plot(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
                ax.plot(exp_data[1][b][:,0]+KGK_end-KMKp_beg,exp_data[1][b][:,1],color='b',marker='*')
            #
            ax.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='*',label='new symm' if b == 0 else '')
            targ = np.argwhere(np.isfinite(reduced_data[b][:,1]))    #select only non-nan values
            ax.plot(reduced_data[b][targ,0],tb_en[b,targ],color='g',marker='^',ls='-',label='fit' if b == 0 else '')
            ax.plot(reduced_data[b][targ,0],tb_en2[b,targ],color='k',marker='s',ls='-',label='DFT' if b == 0 else '')
        #
        ax.set_xlabel(r'$A^{-1}$')
        ax.set_ylabel('E(eV)')
        plt.legend()
        plt.show()
else:
    print("Using SOC parameters of DFT")
    if not Path(SOC_fn).is_file() and save_SOC:
        np.save(SOC_fn,DFT_values[-3:])
    SOC_pars = DFT_values[-3:]

print("Using SOC parameters: ",SOC_pars)

"""
We want a minimization of tb bands vs experiment which penalizes going away from DFT initial values.
"""
print("Computing tb parameters")
#
temp_dn = fs.get_temp_dn(machine,spec_args)
if not Path(temp_dn).is_dir():
    os.system("mkdir "+temp_dn)
#
rand_vals = np.random.rand(DFT_values.shape[0]-3)*0.1+0.95 #random value between 0.95 and 1.05
rand_vals = np.append(rand_vals,np.ones(3))
initial_point_full = DFT_values*rand_vals    #eps,t,off,lam
#
Bounds_full = fs.get_bounds(DFT_values,spec_args)
HSO = cfs.find_HSO(SOC_pars[-2:])
args_chi2 = (reduced_data,HSO,SOC_pars,machine,spec_args,ind_random)
Bounds = Bounds_full[:-2]
initial_point = initial_point_full[:-2]
#
result = minimize(fs.chi2,
        args = args_chi2,
        x0 = initial_point,
        bounds = Bounds,
        method = 'Nelder-Mead',
        options = {
            'disp': False,
            'adaptive' : True,
            'fatol': 1e-4,
#            'xatol': 1e-8,
            'maxiter': 1e6,
            },
        )

min_chi2 = result.fun
print("Minimum chi2: ",min_chi2)
t_final = timedelta(seconds=ttt()-t_initial)
print("Total time: ",str(t_final))





































