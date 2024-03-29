import numpy as np
import functions as fs
import parameters as ps
from pathlib import Path
import os,sys
from scipy.optimize import minimize

machine = fs.get_machine(os.getcwd())

TMD,cuts,range_par = fs.get_parameters(int(sys.argv[1]))
cuts_fn = fs.get_cuts_fn(cuts)

print("Computing TMD: ",TMD,", in cuts: ",cuts_fn," and range: ",range_par)

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,cuts,machine)
if machine == 'loc':    #plot exp to see if they are aligned
    import matplotlib.pyplot as plt
    plt.figure()
    for b in range(2):
        plt.scatter(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
        plt.scatter(exp_data[1][b][:,0]+(exp_data[0][b][-1,0]-exp_data[1][b][0,0]),exp_data[1][b][:,1],color='m',marker='*',label='experiment' if b == 0 else '')
    plt.show()
    input("Continue?")

#Arguments of chi^2 function

initial_point = ps.initial_pt[TMD]  #DFT values
len_pars = len(initial_point)

args_chi2 = (exp_data,TMD,machine,range_par,cuts)
initial_chi2 = fs.chi2(initial_point,*args_chi2)
print("Initial chi2: ",initial_chi2)

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




































