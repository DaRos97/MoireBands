import numpy as np
import functions as fs
import parameters as ps
from pathlib import Path
import os

TMD = 'WSe2'                #Material
#TMD = 'WS2'                #Material
machine = fs.get_machine(os.getcwd())

#Experimental data of monolayer 
#For each material, 2 TVB (because of SO) on the 2 cuts
exp_data = fs.get_exp_data(TMD,machine)

if 0:   #plot exp bands
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for i in range(2):
        for j in range(2):
            plt.plot(exp_data[i][j][:,0],exp_data[i][j][:,1],'*',label=str(i))
    plt.show()
    exit()

#Arguments of chi^2 function
args_chi2 = (exp_data,TMD,machine,False)

initial_point = ps.initial_pt[TMD]  #DFT values
len_pars = len(initial_point)

initial_chi2 = fs.chi2(initial_point,*args_chi2)

print("Initial chi2: ",initial_chi2)

if 0: #Single parameter search
    n_attempts = 7
    range_par = 0.1     #10%
    arr_chi2 = np.zeros((len_pars,n_attempts))
    for i in range(len_pars):
        for j in range(n_attempts):
            if j == n_attempts//2:
                arr_chi2[i,j] = 1e5
                continue
            new_pars = np.copy(initial_point)
            new_pars[i] = new_pars[i] + (j-n_attempts//2)/(n_attempts//2)*new_pars[i]*range_par
            arr_chi2[i,j] = fs.chi2(new_pars,*args_chi2)
            print(i,j)

    argmin = np.argmin(arr_chi2)
    i_ = argmin//n_attempts
    j_ = argmin%n_attempts

    print(argmin,i_,j_,arr_chi2[i_,j_])
    final_par = np.copy(initial_point)
    final_par[i_] = final_par[i_] + (j_-n_attempts//2)/(n_attempts//2)*final_par[i_]*range_par
    args_chi2 = (exp_data,TMD,machine,True)
    fs.chi2(final_par,*args_chi2)

if 1:   #minimization
    from scipy.optimize import minimize
    range_par = 0.1     #10%
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
    fit_fn = fs.get_fit_fn(range_par,TMD,result.fun,machine)
    np.save(fit_fn,final_pars)
    if machine == 'loc':
        args_chi2 = (exp_data,TMD,machine,True)
    fs.chi2(final_pars,*args_chi2)




































