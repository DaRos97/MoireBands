"""Description of the script:
py monolayer.py arg1
arg1: index of specification arguments, which includes
    - M -> material
    - P -> parameter of chi2_1
    - rp -> bounds of energies and hoppings
    - rl -> bounds of SOC

Description of the code:
We extract the experimental data, adjust it with an offset (and symmetrize it), finally we take a subset of points to speed up the computation.
We extract the DFT parameters and compute initial point and the bounds.
Compute the chi squared function by evaluating the bands an minimizing it within the bounds.
"""
import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import utils
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
machine = cfs.get_machine(os.getcwd())

disp = True                                     #Display messages during computation
max_eval = 5e6                                  #max number of chi2 evaluations

if len(sys.argv) != 2:
    print("Usage: py mmain.py arg1",
          "\narg1: index of parameter list")
    exit()

""" Import args for minimization """
argc = int(sys.argv[1])
if machine == 'maf':
    argc -= 1
args_minimization = utils.get_args(argc)
TMD = args_minimization[0]
pts = args_minimization[-1]

""" Import experimental data of monolayer """
data = cfs.monolayerData(TMD)

if disp:
    print("------------CHOSEN PARAMETERS------------")
    print(" TMD: ",args_minimization[0],
          "\n K_1: ","{:.4f}".format(args_minimization[1]),
          "\n K_2: ","{:.4f}".format(args_minimization[2]),
          "\n K_3: ","{:.4f}".format(args_minimization[3]),
          "\n K_4: ","{:.4f}".format(args_minimization[4]),
          "\n Bound pars: ","{:.2f}".format(args_minimization[5]*100)+"%",
          "\n Bound z-pars: ","{:.2f}".format(args_minimization[6]*100)+"%",
          "\n Bound xy-pars: ","{:.2f}".format(args_minimization[7]*100)+"%",
          "\n Bounds SOC: ","{:.2f}".format(args_minimization[8]*100)+"%",
          )
    print(" Using ",pts," points of interpolated data.")

""" Fitting """
DFT_values = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC
Bounds_full = fsm.get_bounds(DFT_values,args)
if args_minimization[8]==0:     # SOC bounds set to 0
    print("Fitting only tb (excluding SOC)")
    HSO = cfs.find_HSO(DFT_values[-2:])
    args_chi2 = (data,HSO,DFT_values[-2:],machine,args_minimization,max_eval)
    Bounds = Bounds_full[:-2]
    initial_point = DFT_values[:-2]
    func = fsm.chi2
else:
    print("Fitting all parameters")
    args_chi2 = (data,machine,args_minimization,max_eval)
    Bounds = Bounds_full
    initial_point = DFT_values
    func = fsm.chi2_full

result = minimize(func,
    args = args_chi2,
    x0 = initial_point,
    bounds = Bounds,
    method = 'Nelder-Mead',
    options = {
        'disp': True,
        'adaptive' : True,
        'fatol': 1e-4,
#            'xatol': 1e-8,
        'maxiter': 1e6,
        },
    )

""" Plotting results """
HSO = cfs.find_HSO(result.x[-2:])
print("Minimization finished with optimal chi2: %.4f"%result.fun)
print("Plotting results")
full_pars = np.append(result.x,off_SOC_pars) if fit_off_SOC_separately else result.x
best_en = cfs.energy(full_pars,HSO,data,args_minimization[0])
fsm.plotResults(full_pars,best_en,data,args_minimization,machine,result.fun)

else:   # To just plot results if they weren't
    import os, glob
    home_dn = fsm.get_home_dn(machine)
    temp_dn = cfs.getFilename(('temp',*args_minimization),dirname=home_dn+'Data/')+'/'
    npy_files = glob.glob(os.path.join(temp_dn, "*.npy"))
    full_pars = np.load(npy_files[0])
    HSO = cfs.find_HSO(full_pars[-2:])

    best_en = cfs.energy(full_pars,HSO,data,args_minimization[0])
    fsm.plotResults(full_pars,best_en,data,args_minimization,machine,float(npy_files[0][-10:-4]))




































