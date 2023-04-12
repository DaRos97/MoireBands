import numpy as np
import sys
import getopt
#
from scipy.optimize import differential_evolution as D_E
#
import scipy.linalg as la
from time import time as tt
#
import functions as fs
import parameters as ps
from input_monolayer import input_data
from input_monolayer import input_data

####not in cluster
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

dirname = "../Data/11_bands/"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "M:")
    M = 'WSe2'               #Material
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-M']:
        M = arg
#
a_mono = ps.dic_params_a_mono[M]

result = D_E(fs.chi2,
    bounds = ps.bounds[M],
    args = (input_data[M],M,a_mono),
    maxiter = 1000,
    popsize = 15,
    tol = 0.01,
    disp = True,
    updating = 'deferred',      #'immediate'
    workers = 2,
    x0 = ps.initial_pt[M],
    )

final_pars = np.array(result.x)
k_pts, final_en = fs.final_energies(final_pars,M,a_mono)
par_filename = dirname + 'fit_pars_'+M+'.npy'
np.save(par_filename,final_pars)


plt.figure()
plt.subplot(1,2,1)
plt.plot(input_data[M][0,0],input_data[M][0,1],'g*')
plt.plot(k_pts,final_en[0],'r-')

plt.subplot(1,2,2)
plt.plot(input_data[M][1,0],input_data[M][1,1],'g*')
plt.plot(k_pts,final_en[1],'r-')

plt.show()

