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
import functions_monolayer as fsm
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
machine = cfs.get_machine(os.getcwd())

disp = True                                     #Display messages during computation
fit_off_SOC_separately = 0#False                    #Fit (offset and) SOC separately from tb parameters
plot_off_SOC_fit = 0#False
max_eval = 1e6              #max number of chi2 evaluations

if len(sys.argv) != 2:
    print("Usage: py monolayer.py arg1",
          "\narg1: index of parameter list")
    exit()

# Import spec_args
argc = int(sys.argv[1])
if machine == 'maf':
    argc -= 1
spec_args = fsm.get_spec_args(argc)
TMD = spec_args[0]
ptsPerPath = spec_args[-1]

# Import experimental data of monolayer 
dataObject = cfs.dataWS2() if TMD=="WS2" else cfs.dataWSe2()
data = dataObject.getFitData(ptsPerPath)

if disp:
    print("------------CHOSEN PARAMETERS------------")
    print(" TMD: ",spec_args[0],
          "\n chi2_1 parameter: ","{:.4f}".format(spec_args[1]),
          "\n Bound parameters: ","{:.2f}".format(spec_args[2]*100)+"%",
          "\n Bounds SOC: ","{:.2f}".format(spec_args[3]*100)+"%",
          #"\n Index random evaluation: ",ind_random
          )
    print(" Using ",ptsPerPath," points, for a total of ",data.shape[0]," points")

# Import DFT values of tb parameters
DFT_values = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC

"""
We start by computing offset and SOC parameters by fitting the energy of the 2 top bands at Gamma and K.
"""
off_SOC_fn = fsm.get_SOC_fn(TMD,machine)
if fit_off_SOC_separately:
    print("\nFitting offset and SOC at Gamma and K.")
    args_chi2_off_SOC = (reduced_data, DFT_values[:-3], spec_args[0], machine)
    if not Path(off_SOC_fn).is_file():
        initial_point_off_SOC = DFT_values[-3:]
        lb = -10#0.5  #lower bound (%)
        ub = 10#1.5  #upper bound (%)
        Bounds_off_SOC = ((DFT_values[-3]*ub,DFT_values[-3]*lb),
                          (DFT_values[-2]*lb,DFT_values[-2]*ub),
                          (DFT_values[-1]*lb,DFT_values[-1]*ub))
        result_off_SOC = minimize(
                fsm.chi2_off_SOC,
                args = args_chi2_off_SOC,
                x0 = initial_point_off_SOC,
                bounds = Bounds_off_SOC,
                method = 'Nelder-Mead',
                options = {
                    'disp': disp,
                    'adaptive' : False,
                    'fatol': 1e-6,
                    'xatol': 1e-8,
                    'maxiter': 1e6,
                    },
                )
        off_SOC_pars = result_off_SOC.x
        if disp:
            print("Result: Offset, SOC "+TMD[0]+", SOC "+TMD[1:-1])
            print(DFT_values[-3:])
            print("-->")
            print(off_SOC_pars)
            print("Chi^2 distance: ",fsm.chi2_SOC(off_SOC_pars,*args_chi2_off_SOC))
    #
    if plot_off_SOC_fit:    #Plot result
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1,1,1)
        KGK_end = exp_data[0][0][-1,0]
        KMKp_beg = exp_data[1][0][0,0]
        ikl = exp_data[0][0].shape[0]//2+1
        HSO_new = cfs.find_HSO(off_SOC_pars[-2:])
        HSO_old = cfs.find_HSO(DFT_values[-2:])
        full_pars = np.append(DFT_values[:-3],off_SOC_pars)
        tb_en_new = cfs.energy(full_pars,HSO_new,reduced_data,spec_args[0])
        tb_en_old = cfs.energy(DFT_values,HSO_old,reduced_data,spec_args[0])
        for b in range(2):
            ax.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='r',marker='*',label='experiment' if b == 0 else '')
            targ = np.argwhere(np.isfinite(reduced_data[b][:,1]))    #select only non-nan values
            ax.plot(reduced_data[b][targ,0],tb_en_new[b,targ],color='g',marker='^',ls='-',label='fit' if b == 0 else '')
            ax.plot(reduced_data[b][targ,0],tb_en_old[b,targ],color='k',marker='s',ls='-',label='DFT' if b == 0 else '')
        #
        ax.set_xlabel(r'$A^{-1}$')
        ax.set_ylabel('E(eV)')
        plt.legend()
        plt.show()
        exit()
else:
    off_SOC_pars = DFT_values[-3:]
    print("Using offset and SOC parameters of DFT: ",off_SOC_pars)

"""
Here we fit the rest of the parameters.
We want a minimization of tb bands vs experiment which penalizes going away from DFT initial values.
"""
if disp:
    txt_SOC_fit = 'NOT including' if fit_off_SOC_separately else "including"
    print("Fitting of tb parameters "+txt_SOC_fit+" SOC parameters")
#
#
#rand_vals = np.random.rand(DFT_values.shape[0]-3)*0.1+0.95 #random value between 0.95 and 1.05
#rand_vals = np.append(rand_vals,np.ones(3))
initial_point_full = np.append(DFT_values[:-3],off_SOC_pars)#*rand_vals    #eps,t,off,lam
#
Bounds_full = fsm.get_bounds(DFT_values,spec_args)
if fit_off_SOC_separately or spec_args[3]==0:
    HSO = cfs.find_HSO(off_SOC_pars[-2:])
    args_chi2 = (data,HSO,off_SOC_pars[-2:],machine,spec_args,max_eval)
    Bounds = Bounds_full[:-2]
    initial_point = initial_point_full[:-2]
    func = fsm.chi2_tb
else:
    args_chi2 = (data,machine,spec_args,max_eval)
    Bounds = Bounds_full
    initial_point = initial_point_full
    func = fsm.chi2_full
#
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

min_chi2 = result.fun
print("Minimization finished with optimal chi2: %.4f"%min_chi2)





































