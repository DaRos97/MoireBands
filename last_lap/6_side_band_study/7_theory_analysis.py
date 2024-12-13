"""Here we compute a theta vs k-distance plot, independent of the experimental measurements"""

import numpy as np
import matplotlib.pyplot as plt
import Functions_7 as fs

n_th = 20
list_th = np.linspace(0,5,n_th)
n_V = 5
list_V = np.linspace(0.1,0.5,n_V)
n_sb = 3        #number of side bands
dist_k = np.zeros((n_th,n_V,n_sb))
for ith in range(n_th):
    for iV in range(n_V):
        th = list_th[ith]
        V = list_V[iV]
        dist_k[ith,iV] = fs.compute_sb_distance(th,V,n_sb)

