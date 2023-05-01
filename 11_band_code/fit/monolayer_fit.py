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

dirname = "../../Data/11_bands/"
#dirname = "/home/users/r/rossid/0_MOIRE/Data/"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "M:",["pts=","cpu=","plot","final"])
    M = 'WSe2'               #Material
    considered_pts = -1
    n_cpu = 1
    plot = False
    final = False
    save = True 
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-M']:
        M = arg
    if opt == '--pts':
        considered_pts = int(arg)
    if opt == '--cpu':
        n_cpu = int(arg)
    if opt == '--plot':
        plot = True
    if opt == '--final':
        final = True
if plot:
    n_cpu = 1
#Monolayer lattice length
a_mono = ps.dic_params_a_mono[M]
#Data
filename1 = 'input_data/KGK_'+M+'_band1_v1.txt'
filename2 = 'input_data/KGK_'+M+'_band2_v1.txt'
input_data_full = [fs.convert(filename1),fs.convert(filename2)]
if not (input_data_full[0][:,0] == input_data_full[1][:,0]).all():
    print("k-pts different in two points, code not valid")
    exit()
#Number of points to consider
N = len(input_data_full[0][:,0])
if considered_pts < 0 or final:
    considered_pts = N
new_N = N//(N//considered_pts)
#print("Points in input data: ",N)
#print("Considering ",new_N," for each band in the fit")
input_data = fs.reduce_input(input_data_full,considered_pts) 
input_energies = [input_data[0][:,1],input_data[1][:,1]]
#k points in path
k_pts_scalar = input_data[0][:,0]
k_pts_vec = fs.find_vec_k(k_pts_scalar,'KGC')
#Arguments of chi^2 function
args_chi2 = (input_energies,M,a_mono,new_N,k_pts_vec)
#Initial point for minimization
if final:       #use final saved values
    temp_filename = dirname+'fit_pars_'+M+'.npy'
else:           #try to use temp saved values
    temp_filename = 'temp_fit_pars_'+M+'.npy'
if Path(temp_filename).is_file():
    initial_point = np.load(temp_filename)
#    print("Using previously saved initial point")
else:
    initial_point = ps.initial_pt[M]
#    print("Starting from DFT values")
#Evaluate initial value of chi^2
initial_chi2 = fs.chi2(initial_point,*args_chi2)
#print("Initial chi2 is ",initial_chi2)
if plot:# or final:
    ens = fs.energies(initial_point,M,a_mono,k_pts_vec)
    plt.figure(figsize=(15,8))
    plt.suptitle(M)
    plt.subplot(1,2,1)
    plt.plot(k_pts_scalar,ens[0],'r-')
    plt.plot(k_pts_scalar,input_energies[0],'g*',zorder=-1)
    plt.subplot(1,2,2)
    plt.plot(k_pts_scalar,ens[1],'r-')
    plt.plot(k_pts_scalar,input_energies[1],'g*',zorder=-1)
    plt.show()
if final: #print k and energy -> to external output
    final_en = fs.energies(initial_point,M,a_mono,k_pts_vec)
    print("TMD:\t",M,'\n')
    for b in range(2):
        print("band ",str(b+1))
        for i in range(len(k_pts_scalar)):
            print("{:.8f}".format(k_pts_scalar[i]),'\t',"{:.8f}".format(final_en[b,i]))
    exit()
#Bounds
Bounds = []
rg = 0.5        #proportional bound around initial values
rg2 = 0.1   #bound irrespective of parameter value
rg_L = 0.1           #bound spin orbit (proportional)
list_SO = [40,41]        #indexes of SO coupling terms in paramter space
for i,p in enumerate(initial_point):
    pp = np.abs(p)
    if i in list_SO:
        Bounds.append((p-pp*rg_L,p+pp*rg_L))
    else:
        Bounds.append((p-pp*rg-rg2,p+pp*rg+rg2))
#Minimization
result = D_E(fs.chi2,
    bounds = Bounds,
    args = args_chi2,
    maxiter = 1000,
    popsize = 15,
    tol = 0.01,
    disp = True,
    workers = n_cpu,
    updating = 'deferred' if n_cpu != 1 else 'immediate',
    x0 = initial_point
    )

final_pars = np.array(result.x)
if plot:
    final_en = fs.energies(final_pars,M,a_mono,k_pts_vec)
    plt.figure()
    plt.plot(k_pts_scalar,input_energies[0],'g*')
    plt.plot(k_pts_scalar,input_energies[1],'g*')
    plt.plot(k_pts_scalar,final_en[0],'r-')
    plt.plot(k_pts_scalar,final_en[1],'r-')
    plt.show()
if save:
#    print("saving final values")
    par_filename = dirname + 'fit_pars_'+M+'.npy'
    np.save(par_filename,final_pars)




