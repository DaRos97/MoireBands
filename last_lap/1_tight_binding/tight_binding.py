import numpy as np
import functions as fs
import parameters as ps
from pathlib import Path
import os,sys
from scipy.optimize import minimize

machine = fs.get_machine(os.getcwd())

TMD,fixed_SO,range_par = fs.get_parameters(int(sys.argv[1]))

print("Computing TMD: ",TMD,", fixed SO: ",fixed_SO," and range: ",range_par)

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,machine)
if 0 and machine == 'loc':    #plot exp to see if they are aligned
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14,7))
    plt.title(TMD)
    for b in range(2):
        plt.scatter(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
        plt.scatter(exp_data[1][b][:,0]+(exp_data[0][b][-1,0]-exp_data[1][b][0,0]),exp_data[1][b][:,1],color='m',marker='*',label='experiment' if b == 0 else '')
    plt.xlabel(r'$A^{-1}$')
    plt.ylabel('E(eV)')
    plt.show()
    input("Continue?")

#Arguments of chi^2 function

DFT_values = ps.initial_pt[TMD]  #DFT values
initial_point = DFT_values[:-2] if fixed_SO else DFT_values

len_pars = len(initial_point)

initial_point=np.array(initial_point)
print(len_pars,min(abs(initial_point[:7])),max(abs(initial_point[:7])))
print(len_pars,min(abs(initial_point[7:])),max(abs(initial_point[7:])))
exit()

args_chi2 = (exp_data,TMD,machine,range_par,fixed_SO,DFT_values[-2:]) 
initial_chi2 = fs.chi2(initial_point,*args_chi2)
print("Initial chi2: ",initial_chi2)

nn = len_pars-1 if fixed_SO else len_pars-3
Bounds = []
for i in range(len_pars):     #tb parameters
    temp_1 = initial_point[i]*(1-range_par)
    temp_2 = initial_point[i]*(1+range_par)
    if initial_point[i]<0:
        temp = (temp_2,temp_1)
    else:
        temp = (temp_1,temp_2)
    if i == nn:
        Bounds.append((-3,0))
    else:
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
fit_fn = fs.get_fit_fn(range_par,TMD,result.fun,fixed_SO,machine)
np.save(fit_fn,final_pars)




































