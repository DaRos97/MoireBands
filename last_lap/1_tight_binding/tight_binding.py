import numpy as np
import functions as fs
import parameters as ps
from pathlib import Path
import os,sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt

machine = fs.get_machine(os.getcwd())

#TMD,range_par = fs.get_parameters(int(sys.argv[1]))
TMD = 'WSe2'

type_bound = 'fixed'

range_dic = {'fixed': 0.1, 'relative': 0.5}
range_par = range_dic[type_bound]

print("Computing TMD: ",TMD," and range: ",range_par," ",type_bound)

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,machine)
if 0 and machine == 'loc':    #plot exp to see if they are aligned
    plt.figure(figsize=(14,7))
    plt.title(TMD)
    for b in range(2):
        plt.scatter(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
        plt.scatter(exp_data[1][b][:,0]+(exp_data[0][b][-1,0]-exp_data[1][b][0,0]),exp_data[1][b][:,1],color='m',marker='*',label='experiment' if b == 0 else '')
    plt.xlabel(r'$A^{-1}$')
    plt.ylabel('E(eV)')
    plt.show()
    exit()

#Arguments of chi^2 function
DFT_values = ps.initial_pt[TMD]  #DFT values
initial_point = np.array(DFT_values[:-2]) #not the SO values
len_pars = initial_point.shape[0]
args_chi2 = (exp_data,TMD,machine,range_par,type_bound,fs.find_HSO(DFT_values[-2:]))

#
initial_chi2 = fs.chi2(initial_point,*args_chi2)
print("Initial chi2: ",initial_chi2)

Bounds = fs.get_bounds(initial_point,range_par,type_bound)

if 1:
    """
    We want a minimization of tb bands vs experiment which penalizes going away from DFT initial values.
    """
    result = minimize(fs.pen_chi2,
            args = args_chi2,
            x0 = initial_point,
            bounds = Bounds,
            method = 'Nelder-Mead',
            options = {
                'disp': True if machine=='loc' else False,
                'adaptive' : True,
                'fatol': 1e-8,
                'xatol': 1e-8,
                'maxiter': 1e6,
                },
            )
    min_chi2 = result.fun
    print("Minimum chi2: ",min_chi2)
else:
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
                'maxiter': 1e6,
                },
            )
    min_chi2 = result.fun
    print("Minimum chi2: ",min_chi2)


final_pars = np.array(result.x)
fit_fn = fs.get_fit_fn(TMD,range_par,ty,min_chi2,machine)
np.save(fit_fn,final_pars)




































