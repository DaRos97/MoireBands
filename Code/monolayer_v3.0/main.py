"""
This time we fix the orbital character at G and K to the DFT-derived one.
We keep as constraint the orbital character at M.
"""
import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code/'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code/'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import utils
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
machine = cfs.get_machine(os.getcwd())

disp = machine=='loc'                                     #Display messages during computation
max_eval = 5e6                                  #max number of chi2 evaluations

""" Import args for minimization """
if len(sys.argv) != 3:
    print("Usage: python main.py arg1 arg2",
          "\narg1: TMD, arg2: index of parameter list")
    exit()
TMD = sys.argv[1]
if TMD not in ['WSe2','WS2']:
    raise ValueError("TMD not recognized: ",TMD)
argc = int(sys.argv[2])
if machine == 'maf':
    argc -= 1
args_minimization = utils.get_args(TMD,argc)
pts = args_minimization['pts']

""" Import experimental data of monolayer """
data = cfs.monolayerData(TMD,master_folder,pts=pts)

if disp:
    print("------------CHOSEN PARAMETERS------------")
    print(" TMD: ",TMD,
          "\n K_1: ","{:.6f}".format(args_minimization['Ks'][0]),
          "\n K_2: ","{:.6f}".format(args_minimization['Ks'][1]),
          "\n K_3: ","{:.6f}".format(args_minimization['Ks'][2]),
          "\n K_4: ","{:.6f}".format(args_minimization['Ks'][3]),
          "\n K_5: ","{:.6f}".format(args_minimization['Ks'][4]),
          "\n K_5: ","{:.6f}".format(args_minimization['Ks'][5])
    )
    if args_minimization['boundType']=='relative':
        print(
            "\n Bound general pars: %.2f"%(args_minimization['Bs'][0]*100)+"%",
            "\n Bound z-pars:       %.2f"%(args_minimization['Bs'][1]*100)+"%",
            "\n Bound xy-pars:      %.2f"%(args_minimization['Bs'][2]*100)+"%",
            "\n Bound SOC:          %.2f"%(args_minimization['Bs'][3]*100)+"%"
        )
    elif args_minimization['boundType']=='absolute':
        print(
            " Bound epsilon: (-%d,%d)"%(args_minimization['Bs'][0],args_minimization['Bs'][0]),
            "\n Bound t1:      (-%d,%d)"%(args_minimization['Bs'][1],args_minimization['Bs'][1]),
            "\n Bound t5:      (-%d,%d)"%(args_minimization['Bs'][2],args_minimization['Bs'][2]),
            "\n Bound t6:      (-%d,%d)"%(args_minimization['Bs'][3],args_minimization['Bs'][3]),
            "\n Bound SOC:     (-%d,%d)"%(args_minimization['Bs'][4],args_minimization['Bs'][4]),
        )
    print(" Using ",pts," points of interpolated data.")
    print("-"*15)

""" Fitting """
DFT_values = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC
Bounds_full = utils.get_bounds(DFT_values,args_minimization)
if args_minimization['Bs'][-1]==0:     # SOC bounds set to 0
    print("Fitting only tb (excluding SOC)")
    print("-"*15)
    HSO = cfs.find_HSO(DFT_values[-2:])
    args_chi2 = (data,HSO,DFT_values[-2:],machine,args_minimization,max_eval,False)
    Bounds = Bounds_full[:-2]
    initial_point = DFT_values[:-2]
    func = utils.chi2
else:
    print("Fitting all parameters")
    print("-"*15)
    args_chi2 = (data,machine,args_minimization,max_eval,False)
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
        'maxiter': 1e6,
        },
    )
final_pars = result.x
resultChi2 = result.fun

print("Final chi2: ",resultChi2)
print("Final parameters: ",final_pars)


































