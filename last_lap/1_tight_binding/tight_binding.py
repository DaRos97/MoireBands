import numpy as np
import functions as fs
import parameters as ps
from pathlib import Path
import os,sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt

machine = fs.get_machine(os.getcwd())
ss = 1 if machine == 'maf' else 0   #Jobs in mafalda start from 1

ind = 0 if len(sys.argv)==1 else int(sys.argv[1])-ss
#ind labels the random initialization, we only do for WSe2 to start
TMD = fs.TMDs[0]

P = 1.0
rp = 1.0
rl = 0.1
spec_args = (P,rp,rl)

print("Computing TMD: ",TMD," with parameters: ",spec_args)

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,machine)
symm_data = fs.get_symm_data(exp_data)
if 0 and machine == 'loc':    
    #plot exp to see if they are aligned
    plt.figure(figsize=(14,7))
    plt.title(TMD)
    for b in range(2):
        plt.scatter(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
        plt.scatter(exp_data[1][b][:,0]+(exp_data[0][b][-1,0]-exp_data[1][b][0,0]),exp_data[1][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
        plt.scatter(symm_data[b][:,0],symm_data[b][:,1],color='r',label='new symm')
    #
    plt.xlabel(r'$A^{-1}$')
    plt.ylabel('E(eV)')
    plt.legend()
    plt.show()
    exit()

#Arguments of chi^2 function
DFT_values = ps.initial_pt[TMD]  #DFT values
rand_vals = np.random.rand(len(DFT_values)-3)*0.1+0.95 #random value between 0.95 and 1.05
rand_vals = np.append(rand_vals,np.ones(3))
initial_point = np.array(DFT_values)*rand_vals    #t,eps,lam,off
len_pars = initial_point.shape[0]
args_chi2 = (symm_data,TMD,machine,spec_args,ind)
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
            'disp': True if machine=='loc' else False,
            'adaptive' : True,
            'fatol': 1e-8,
            'xatol': 1e-8,
            'maxiter': 1e8,
            },
        )
min_chi2 = result.fun
print("Minimum chi2: ",min_chi2)


final_pars = np.array(result.x)
fit_fn = fs.get_fit_fn(TMD,spec_args,min_chi2,ind,machine)
np.save(fit_fn,final_pars)




































