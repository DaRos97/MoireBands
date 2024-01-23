import numpy as np
import functions as fs
import parameters as ps
from pathlib import Path
import os,sys

machine = fs.get_machine(os.getcwd())

TMD,cuts,range_par = fs.get_parameters(int(sys.argv[1]))
cuts_fn = ''
for i in range(len(cuts)):
    cuts_fn += cuts[i]
    if i != len(cuts)-1:
        cuts_fn += '_'
print("Computing TMD: ",TMD,", in cuts: ",cuts_fn," and range: ",range_par)
exit()

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,cuts,machine)

#Arguments of chi^2 function

initial_point = ps.initial_pt[TMD]  #DFT values
len_pars = len(initial_point)

from scipy.optimize import minimize
args_chi2 = (exp_data,TMD,machine,range_par,cuts,False)
Bounds = []
for i in range(len_pars):
    temp_1 = initial_point[i]*(1-range_par)
    temp_2 = initial_point[i]*(1+range_par)
    if initial_point[i]<0:
        temp = (temp_2,temp_1)
    else:
        temp = (temp_1,temp_2)
    Bounds.append(temp)
#
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
print("Minimum chi2: ",result.fun)
final_pars = np.array(result.x)
fit_fn = fs.get_fit_fn(range_par,TMD,result.fun,cuts,machine)
np.save(fit_fn,final_pars)




































