import numpy as np
import sys
import getopt
from pathlib import Path
#
from scipy.optimize import differential_evolution as D_E        #try a gradient descent
#
import scipy.linalg as la
from time import time as tt
#
import functions as fs
import parameters as ps

####not in cluster
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import os

dirname = "../../Data/11_bands/"
#dirname = "/home/users/r/rossid/0_MOIRE/Data/"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "M:",["pts=","cpu=","final","SO"])
    M = 'WSe2'               #Material
    input_considered_pts = -1
    n_cpu = 1
    final = False
    save = True 
    consider_SO = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-M']:
        M = arg
    if opt == '--pts':
        input_considered_pts = int(arg)
    if opt == '--cpu':
        n_cpu = int(arg)
    if opt == '--final':
        final = True
    if opt == '--SO':
        consider_SO = True
txt_SO = "SO" if consider_SO else "noSO"
#Monolayer lattice length
a_mono = ps.dic_params_a_mono[M]
#Data
paths = ['KGK','KMKp']
cuts = len(paths)
input_energies = []
k_pts_vec = []
k_pts_scalar = []
new_N = []
for P in paths:
    input_data_full = fs.convert(P,M)
    #Number of points to consider
    N = len(input_data_full[0][:,0])        #Full length
    if input_considered_pts < 0 or input_considered_pts > N or final:
        considered_pts = N
    else:
        considered_pts = input_considered_pts
    new_N.append(N//(N//considered_pts))
    input_data = fs.reduce_input(input_data_full,considered_pts) 
    input_energies.append([input_data[0][:,1],input_data[1][:,1]])
    #k points in path
    k_pts_scalar.append(input_data[0][:,0])
    k_pts_vec.append(fs.find_vec_k(k_pts_scalar[-1],P,a_mono))
#Arguments of chi^2 function
SO_pars = [0,0] if consider_SO else ps.initial_pt[M][40:42]
args_chi2 = (input_energies,M,a_mono,new_N,k_pts_vec,SO_pars)
#Initial point for minimization
if final:       #use final saved values
    temp_filename = dirname+'fit_pars_'+M+"_"+txt_SO+'.npy'
else:           #try to use temp saved values
    temp_filename = 'temp_fit_pars_'+M+"_"+txt_SO+'.npy'
if Path(temp_filename).is_file():
    initial_point = np.load(temp_filename)
    print("using saved file")
else:
    initial_point = fs.dft_values(ps.initial_pt[M],consider_SO)
    print("using DFT file")
#Evaluate initial value of chi^2
initial_chi2 = fs.chi2(initial_point,*args_chi2)
if final:
    print("Final chi2 is ",initial_chi2)
    if consider_SO:
        pars_final = initial_point 
    else:
        pars_final = list(initial_point[:-1])
        pars_final.append(SO_pars[0])
        pars_final.append(SO_pars[1])
        pars_final.append(initial_point[-1])
    final_en = fs.energies(pars_final,M,a_mono,k_pts_vec)
    plt.figure(figsize=(15,8))
    plt.suptitle(M)
    for cut in range(cuts):
        plt.subplot(cuts,2,2*cut+1)
        plt.plot(k_pts_scalar[cut],final_en[cut][0],'r-')
        plt.plot(k_pts_scalar[cut],input_energies[cut][0],'g*',zorder=-1)
        plt.subplot(cuts,2,2*cut+2)
        plt.plot(k_pts_scalar[cut],final_en[cut][1],'r-')
        plt.plot(k_pts_scalar[cut],input_energies[cut][1],'g*',zorder=-1)
    plt.show()
    if input("Save k_en list and DFT vs TB table? (y/n)") == 'y':
        #Print k and energy -> to external output AND create table of differences between DFT and fit
        for cut in range(cuts):
            filename_ek = 'result/k_en_'+M+'_'+paths[cut]+'.txt'
            with open(filename_ek, 'w') as f:
                with redirect_stdout(f):
                    print("TMD:\t",M,'\n')
                    for b in range(2):
                        print("band ",str(b+1))
                        for i in range(len(k_pts_scalar[cut])):
                            print("{:.8f}".format(k_pts_scalar[cut][i]),'\t',"{:.8f}".format(final_en[cut][b,i]))
        command = 'python distance_dft.py '+M+' '+str(consider_SO)
        os.system(command)
    exit()
#Bounds
Bounds = []
rg = 1        #proportional bound around initial values
rg2 = 0.1   #bound irrespective of parameter value
for i,p in enumerate(initial_point):
    pp = np.abs(p)
    Bounds.append((p-pp*rg-rg2,p+pp*rg+rg2))
#Minimization
result = D_E(fs.chi2,
    bounds = Bounds,
    args = args_chi2,
    maxiter = 1000,
    popsize = 15,
    tol = 0.01,
#    disp = True,
    workers = n_cpu,
    updating = 'deferred' if n_cpu != 1 else 'immediate',
    x0 = initial_point
    )

final_pars = np.array(result.x)
if save:
    print("Saving with final value: ",result.fun)
    par_filename = dirname + 'fit_pars_'+M+'_'+txt_SO+'.npy'
    np.save(par_filename,final_pars)




