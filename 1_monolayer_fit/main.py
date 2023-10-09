import numpy as np
import sys
import getopt
from pathlib import Path
#
from scipy.optimize import differential_evolution as D_E        #stochastic
from scipy.optimize import minimize as MZ        #grad_descent
type_minimization = 'grad_descent'
#
import scipy.linalg as la
#
import functions as fs
import parameters as ps

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "M:",["pts=","cpu=","final","SO","cluster"])
    TMD = 'WSe2'               #Material
    input_considered_pts = -1   #-1 for all of it
    n_cpu = 1                   #cpus in the D_E function
    final = False               #put to True to compute the table with the parameters AFTER the minimization has finished
    cluster = False
    consider_SO = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-M']:
        TMD = arg
    if opt == '--pts':
        input_considered_pts = int(arg)
    if opt == '--cpu':
        n_cpu = int(arg)
    if opt == '--final':
        final = True
    if opt == '--SO':
        consider_SO = True
    if opt == "--cluster":
        cluster = True
#Adjustments
home_dirname = "/home/dario/Desktop/git/MoireBands/1_monolayer_fit/" if not cluster else "/home/users/r/rossid/1_monolayer_fit/"
save_data_dirname = home_dirname + "Data/"
txt_SO = "SO" if consider_SO else "noSO"
save = True                 
#Monolayer lattice length
a_mono = ps.dic_params_a_mono[TMD]
#Data
paths = ['KGK','KMKp']
cuts = len(paths)
input_energies = []
k_pts_vec = []
k_pts_scalar = []
len_data = []
#Extract experimental data for the 2 BZ paths
for P in paths:
    input_data_full = fs.convert(P,TMD,home_dirname)
    #Number of points to consider
    N = len(input_data_full[0][:,0])        #Full length
    len_data.append(N)
    input_energies.append([input_data_full[0][:,1],input_data_full[1][:,1]])
    #k points in path
    k_pts_scalar.append(input_data_full[0][:,0])
    k_pts_vec.append(fs.find_vec_k(k_pts_scalar[-1],P,a_mono))

#Arguments of chi^2 function
SO_pars = [0,0] if consider_SO else ps.initial_pt[TMD][40:42]
args_chi2 = (input_energies,TMD,a_mono,len_data,k_pts_vec,SO_pars,save_data_dirname)
#Filename of parameters
if final:
    fit_pars_filename = save_data_dirname+'fit_pars_'+TMD+"_"+txt_SO+'.npy'
    if Path(fit_pars_filename).is_file():
        print("Using final value of minimization for final evaluation")
    else:
        fit_pars_filename = save_data_dirname+'temp_fit_pars_'+TMD+"_"+txt_SO+'.npy'
        print("Using temp file for final evaluation")
else:
    fit_pars_filename = "/no_res"
#Extract initial point of minimization (or of final result)
if Path(fit_pars_filename).is_file():
    initial_point = np.load(fit_pars_filename)
    print("Extracting saved fit parameters")
    DFT = False
else:
    initial_point = fs.dft_values(ps.initial_pt[TMD],consider_SO)
    print("Using DFT parameters")
    DFT = True
#Evaluate initial value of chi^2
initial_chi2 = fs.chi2(initial_point,*args_chi2)
ps.temp_res = initial_chi2      #initiate comparison parameter
if DFT: #Save DFT value in temp and use it as initial comparison result for the minimization
    par_filename = save_data_dirname + 'temp_fit_pars_'+TMD+'_'+txt_SO+'.npy'
    np.save(par_filename,initial_point)
    print("saving DFT res ",initial_chi2)

if not final:   #Minimization
    #Bounds
    Bounds = []
    rg = 0.5        #proportional bound around initial values
    min_variation = 0.1
    for i,p in enumerate(initial_point):
        pp = np.abs(p)
        if pp*rg > min_variation:
            Bounds.append((p-pp*rg,p+pp*rg))
        else:
            Bounds.append((p-min_variation,p+min_variation))
        if 0:
            print("initial par ",ps.list_names_all[i],": ",p)
            print("bound: ",Bounds[-1],'\n')
    #Minimization
    if type_minimization == 'stochastic':
        result = D_E(fs.chi2,
            bounds = Bounds,
            args = args_chi2,
    #        maxiter = 1000,
    #        popsize = 20,
    #        tol = 0.01,
            disp = False if cluster else True,
            workers = n_cpu,
            updating = 'deferred' if n_cpu != 1 else 'immediate',
            x0 = initial_point
            )
    elif type_minimization == 'grad_descent':
        result = MZ(fs.chi2,
            args = args_chi2,
            x0 = initial_point,
            bounds = Bounds,
            method = 'Nelder-Mead',
            options = {
                'disp': False if cluster else True,
                'adaptive' : True,
                },
            )

    final_pars = np.array(result.x)
    print("Saving with final value: ",result.fun)
    par_filename = save_data_dirname + 'fit_pars_'+TMD+'_'+txt_SO+'.npy'
    np.save(par_filename,final_pars)
else:
    import matplotlib.pyplot as plt
    from contextlib import redirect_stdout
    import os
    #
    print("Final chi2 is ",initial_chi2)
    if consider_SO:
        pars_final = initial_point 
    else:
        pars_final = list(initial_point[:-1])
        pars_final.append(SO_pars[0])
        pars_final.append(SO_pars[1])
        pars_final.append(initial_point[-1])
    final_en = fs.energies(pars_final,TMD,a_mono,k_pts_vec)
    plt.figure(figsize=(15,8))
    plt.suptitle(TMD)
    for cut in range(cuts):
        plt.subplot(cuts,2,2*cut+1)
        plt.title("Cut "+paths[cut]+", band 0")
        plt.plot(k_pts_scalar[cut],final_en[cut][0],'r-')
        plt.plot(k_pts_scalar[cut],input_energies[cut][0],'g*',zorder=-1)
        plt.subplot(cuts,2,2*cut+2)
        plt.title("Cut "+paths[cut]+", band 1")
        plt.plot(k_pts_scalar[cut],final_en[cut][1],'r-')
        plt.plot(k_pts_scalar[cut],input_energies[cut][1],'g*',zorder=-1)
    plt.show()
    if input("Save k_en list and DFT vs TB table? (y/n)") == 'y':
        #Print k and energy -> to external output AND create table of differences between DFT and fit
        for cut in range(cuts):
            filename_ek = save_data_dirname + 'k_en_'+TMD+'_'+paths[cut]+'.txt'
            with open(filename_ek, 'w') as f:
                with redirect_stdout(f):
                    print("TMD:\t",TMD,'\n')
                    for b in range(2):
                        print("band ",str(b+1))
                        for i in range(len(k_pts_scalar[cut])):
                            print("{:.8f}".format(k_pts_scalar[cut][i]),'\t',"{:.8f}".format(final_en[cut][b,i]))
        command = 'python distance_dft.py '+TMD+' '+str(consider_SO)+' '+save_data_dirname
        os.system(command)
    exit()





