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
fit_off_SOC_separately = False                    #Fit (offset and) SOC separately from tb parameters
plot_off_SOC_fit = 0#True
max_eval = 1e7              #max number of chi2 evaluations

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
          "\n P_par: ","{:.4f}".format(spec_args[1]),
          "\n P_bc: ","{:.4f}".format(spec_args[2]),
          "\n P_dk: ","{:.4f}".format(spec_args[3]),
          "\n P_gap: ","{:.4f}".format(spec_args[4]),
          "\n Bound pars: ","{:.2f}".format(spec_args[5]*100)+"%",
          "\n Bound z-pars: ","{:.2f}".format(spec_args[6]*100)+"%",
          "\n Bound xy-pars: ","{:.2f}".format(spec_args[7]*100)+"%",
          "\n Bounds SOC: ","{:.2f}".format(spec_args[8]*100)+"%",
          #"\n Index random evaluation: ",ind_random
          )
    print(" Using ",ptsPerPath," points, for a total of ",data.shape[0]," points")

# Import DFT values of tb parameters
DFT_values = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC
#DFT_values = np.load("Figures/result_WSe2.npy")
#DFT_values = np.load("Figures/result_newyear_0.01_100_20_0.5_0.2_0.2_1_0.2.npy")
if 0:   # Plot bands and orbital of DFT
    HSO = cfs.find_HSO(DFT_values[-2:])
    DFT_en = cfs.energy(DFT_values,HSO,data,spec_args[0])
    fsm.plotResults(DFT_values,DFT_en,data,spec_args,machine,0,show=True)
    exit()

"""
We start by computing offset and SOC parameters by fitting the energy of the 2 top bands at Gamma and K.
"""
off_SOC_fn = fsm.get_SOC_fn(TMD,machine)
if fit_off_SOC_separately:
    print("\nFitting offset and SOC at Gamma and K.")
    args_chi2_off_SOC = (data, DFT_values[:-3], spec_args, machine)
    if not Path(off_SOC_fn).is_file():
        initial_point_off_SOC = DFT_values[-3:]
        lb = -20#0.5  #lower bound (%)
        ub = 20#1.5  #upper bound (%)
        Bounds_off_SOC = ((DFT_values[-3]*ub,DFT_values[-3]*lb),    #offset if negative
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
            print("Chi^2 distance: ",fsm.chi2_off_SOC(off_SOC_pars,*args_chi2_off_SOC))
    #
    if plot_off_SOC_fit:    #Plot result
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot()
        #KGK_end = data[0][0][-1,0]
        #KMKp_beg = data[1][0][0,0]
        #ikl = exp_data[0][0].shape[0]//2+1
        HSO_new = cfs.find_HSO(off_SOC_pars[-2:])
        HSO_old = cfs.find_HSO(DFT_values[-2:])
        full_pars = np.append(DFT_values[:-3],off_SOC_pars)
        tb_en_new = cfs.energy(full_pars,HSO_new,data,spec_args[0])
        tb_en_old = cfs.energy(DFT_values,HSO_old,data,spec_args[0])
        nbands = 4 if TMD=='WSe2' else 2
        for b in range(nbands):
            ax.plot(data[:,0],data[:,3+b],color='r',marker='*',label='experiment' if b == 0 else '')
            targ = np.argwhere(np.isfinite(data[:,3+b]))    #select only non-nan values
            ax.plot(data[targ,0],tb_en_new[b,targ],color='g',marker='^',ls='-',label='fit' if b == 0 else '')
            ax.plot(data[targ,0],tb_en_old[b,targ],color='k',marker='s',ls='-',label='DFT' if b == 0 else '')
        #
        ax.set_xlabel(r'$A^{-1}$')
        ax.set_ylabel('E(eV)')
        plt.legend()
        plt.show()
        exit()
else:
    off_SOC_pars = DFT_values[-3:]
    #print("Using offset and SOC parameters of DFT: ",off_SOC_pars)
    print("Fitting all parameters together")

"""
Here we fit the rest of the parameters.
We want a minimization of tb bands vs experiment which penalizes going away from DFT initial values.
"""
#rand_vals = np.random.rand(DFT_values.shape[0]-3)*0.1+0.95 #random value between 0.95 and 1.05 of initial DFT value
#rand_vals = np.append(rand_vals,np.ones(3))
initial_point_full = np.append(DFT_values[:-3],off_SOC_pars)#*rand_vals    #eps,t,off,lam
#initial_point_full = DFT_values
#
Bounds_full = fsm.get_bounds(initial_point_full,spec_args)
if fit_off_SOC_separately:
    print("Fitting only tb (excluding SOC)")
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
    if 0:
        print("Test run")
        func(initial_point,*args_chi2)
        exit()
#
if 1:
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

    HSO = cfs.find_HSO(result.x[-2:])
    print("Minimization finished with optimal chi2: %.4f"%result.fun)
    print("Plotting results")
    full_pars = np.append(result.x,off_SOC_pars) if fit_off_SOC_separately else result.x
    best_en = cfs.energy(full_pars,HSO,data,spec_args[0])
    fsm.plotResults(full_pars,best_en,data,spec_args,machine,result.fun)

else:   # To just plot results if they weren't
    import os, glob
    home_dn = fsm.get_home_dn(machine)
    temp_dn = cfs.getFilename(('temp',*spec_args),dirname=home_dn+'Data/')+'/'
    npy_files = glob.glob(os.path.join(temp_dn, "*.npy"))
    full_pars = np.load(npy_files[0])
    HSO = cfs.find_HSO(full_pars[-2:])

    best_en = cfs.energy(full_pars,HSO,data,spec_args[0])
    fsm.plotResults(full_pars,best_en,data,spec_args,machine,float(npy_files[0][-10:-4]))




































