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

if len(sys.argv) < 2:
    print("Usage: python main.py arg1 -p",
          "\narg1: index of parameter list\n-p for just plotting existing result")
    exit()

""" Import args for minimization """
argc = int(sys.argv[1])
if machine == 'maf':
    argc -= 1
args_minimization = utils.get_args(argc)
TMD = args_minimization['TMD']
pts = args_minimization['pts']

""" Import experimental data of monolayer """
data = cfs.monolayerData(TMD,pts=pts)

if disp:
    print("------------CHOSEN PARAMETERS------------")
    print(" TMD: ",TMD,
          "\n K_1: ","{:.6f}".format(args_minimization['Ks'][0]),
          "\n K_2: ","{:.6f}".format(args_minimization['Ks'][1]),
          "\n K_3: ","{:.6f}".format(args_minimization['Ks'][2]),
          "\n K_4: ","{:.6f}".format(args_minimization['Ks'][3]),
          "\n K_5: ","{:.6f}".format(args_minimization['Ks'][4]),
          "\n Bound pars: %.2f"%(args_minimization['Bs'][0]*100)+"%",
          "\n Bound z-pars: %.2f"%(args_minimization['Bs'][1]*100)+"%",
          "\n Bound xy-pars: %.2f"%(args_minimization['Bs'][2]*100)+"%",
          "\n Bound SOC: %.2f"%(args_minimization['Bs'][3]*100)+"%",
          )
    print(" Using ",pts," points of interpolated data.")
    print("-"*15)

""" Fitting """
if len(sys.argv)==2:
    print("Fitting parameters")
    print("-"*15)
    DFT_values = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC
    Bounds_full = utils.get_bounds(DFT_values,args_minimization['Bs'])
    if args_minimization['Bs'][3]==0:     # SOC bounds set to 0
        print("Fitting only tb (excluding SOC)")
        HSO = cfs.find_HSO(DFT_values[-2:])
        args_chi2 = (data,HSO,DFT_values[-2:],machine,args_minimization,max_eval)
        Bounds = Bounds_full[:-2]
        initial_point = DFT_values[:-2]
        func = utils.chi2
    else:
        print("Fitting all parameters")
        args_chi2 = (data,machine,args_minimization,max_eval)
        Bounds = Bounds_full
        initial_point = DFT_values
        func = utils.chi2_full

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
    final_pars = result.x
    resultChi2 = result.fun
else:
    if sys.argv[2]=='-p':
        print("Loading parameters")
        print("-"*15)
        home_dn = utils.get_home_dn(machine)
        temp_dn = cfs.getFilename(('temp',*list(args_minimization.values())),dirname=home_dn+'Data/',floatPrecision=10)+'/'
        filenames = os.listdir(temp_dn)
        print(filenames)
        for fn in filenames:
            if fn[-4:]=='.npy':
                final_pars = np.load(temp_dn+fn)
                resultChi2 = float(fn.split('_')[1][:-4])
                break
    else:
        raise ValueError("Unrecognized second argument: %s"%sys.argv[2])

""" Plotting results """
print("Plotting results")
HSO = cfs.find_HSO(final_pars[-2:])
best_en = cfs.energy(final_pars,HSO,data.fit_data,args_minimization['TMD'])
utils.plotResults(final_pars,best_en,data.fit_data,args_minimization,machine,resultChi2)



































