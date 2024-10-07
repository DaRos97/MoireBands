"""Description of the script:
py tight_binding.py arg1 arg2
arg1: index of specification arguments, which includes
    - M -> material
    - P -> parameter of chi2_1
    - rp -> bounds of energies and hoppings
    - rl -> bounds of SOC
arg2: index of random realization within 5% of DFT values

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
import functions as fs
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time import time as ttt
from datetime import timedelta

t_initial = ttt()
machine = cfs.get_machine(os.getcwd())

#ind labels the random initialization
ind_spec_args = 0 if len(sys.argv)==1 else int(sys.argv[1])
ind_random = 0 if len(sys.argv) in [1,2] else int(sys.argv[2])
ind_reduced = 7

spec_args = fs.get_spec_args(ind_spec_args) + (ind_reduced,)
TMD = spec_args[0]

print("Computing parameters: ",spec_args," and ind_random: ",ind_random)

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,machine)
symm_data = fs.get_symm_data(exp_data)
reduced_data = fs.get_reduced_data(symm_data,ind_reduced)
if 0 and machine == 'loc':
    #plot exp to see if they are aligned
    plt.figure(figsize=(14,7))
    plt.title(TMD)
    KGK_end = exp_data[0][0][-1,0]
    KMKp_beg = exp_data[1][0][0,0]
    ikl = exp_data[0][0].shape[0]//2+1
    for b in range(2):
        plt.plot(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
        plt.plot(exp_data[1][b][:,0]+KGK_end-KMKp_beg,exp_data[1][b][:,1],color='b',marker='*')
        #
        plt.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='*',label='new symm')
    #
    plt.xlabel(r'$A^{-1}$')
    plt.ylabel('E(eV)')
    plt.legend()
    plt.show()
    exit()

#Arguments of chi^2 function
DFT_values = cfs.initial_pt[TMD]  #DFT values
rand_vals = np.random.rand(len(DFT_values)-3)*0.1+0.95 #random value between 0.95 and 1.05
rand_vals = np.append(rand_vals,np.ones(3))
initial_point = np.array(DFT_values)*rand_vals    #t,eps,lam,off
len_pars = initial_point.shape[0]
args_chi2 = (reduced_data,exp_data,machine,spec_args,ind_random)
#
temp_dn = fs.get_temp_dn(machine,spec_args)
if not Path(temp_dn).is_dir():
    os.system("mkdir "+temp_dn)
initial_chi2 = fs.chi2(initial_point,*args_chi2)
print("Initial chi2: ",initial_chi2)

Bounds = fs.get_bounds(DFT_values,spec_args)

"""
We want a minimization of tb bands vs experiment which penalizes going away from DFT initial values.
"""
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




































