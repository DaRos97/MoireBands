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
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

dirname = "../../Data/11_bands/"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "M:",["pts=","cpu=","plot"])
    M = 'WSe2'               #Material
    considered_pts = 50
    n_cpu = 1
    plot = False
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
if plot:
    n_cpu = 1
#
a_mono = ps.dic_params_a_mono[M]

filename1 = 'input_data/KGK_'+M+'_band1_v1.txt'
filename2 = 'input_data/KGK_'+M+'_band2_v1.txt'
input_data_full = [fs.convert(filename1),fs.convert(filename2)]
if not (input_data_full[0][:,0] == input_data_full[1][:,0]).all():
    print("k-pts different in two points, code not valid")
    exit()
N = len(input_data_full[0][:,0])
new_N = N//(N//considered_pts)
print("Points in input data: ",N)
print("Considering only ",new_N," for each band in the fit")
input_data = fs.reduce_input(input_data_full,considered_pts) 
input_energies = [input_data[0][:,1],input_data[1][:,1]]
#k points in path
k_pts_scalar = input_data[0][:,0]
k_pts_vec = fs.find_vec_k(k_pts_scalar,'KGC')
#Args
args_chi2 = (input_energies,M,a_mono,new_N,k_pts_vec,k_pts_scalar,plot)
#Initial point
temp_filename = 'temp_fit_pars_'+M+'.npy'
if Path(temp_filename).is_file():
    initial_point = np.load(temp_filename)
    print("Using previous initial point")
else:
    initial_point = ps.initial_pt[M]
    print("Starting from DFT values")
initial_chi2 = fs.chi2(initial_point,*args_chi2)
print("Initial chi2 is ",initial_chi2)
if plot:
    ens = fs.energies(initial_point,M,a_mono,k_pts_vec)
    plt.figure()
    plt.plot(k_pts_scalar,input_energies[0],'g*')
    plt.plot(k_pts_scalar,input_energies[1],'g*')
    plt.plot(k_pts_scalar,ens[0],'r-')
    plt.plot(k_pts_scalar,ens[1],'r-')
    plt.show()
#Bounds
Bounds = []
rg = initial_chi2        #range for bounds around dft values
for p in initial_point:
    pp = np.abs(p)
    Bounds.append((p-pp*rg,p+pp*rg))
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
final_en = fs.energies(final_pars,M,a_mono,k_pts_vec)
plt.figure()
plt.plot(k_pts_scalar,input_energies[0],'g*')
plt.plot(k_pts_scalar,input_energies[1],'g*')
plt.plot(k_pts_scalar,final_en[0],'r-')
plt.plot(k_pts_scalar,final_en[1],'r-')

plt.show()
save = input("Save final parameters? [y->all/n]")
if save != 'n':
    par_filename = dirname + 'fit_pars_'+M+'.npy'
    np.save(par_filename,final_pars)



