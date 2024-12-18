"""Here we try to extract the moire potential at Gamma by looking at the distance between the main band and
the X shape of the side bands below.
We do this by only varying the Vg from 0.005 to 0.02, computing at k=Gamma the distance between the main band
and the lower side band (need to get the weights, at Gamma there is only one that remains nonzero). Finally,
we plot this distance as function of Vg and compare with experiment. In S11 we see around 90 meV.
"""
import sys,os
import numpy as np
import scipy
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions3 as fs3
from pathlib import Path
import matplotlib.pyplot as plt
from time import time
from matplotlib.colors import Normalize

machine = cfs.get_machine(cwd)
if machine=='loc':
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm
#
list_Vg = np.linspace(0.005,0.02,15)
monolayer_type, interlayer_symmetry, Vg, Vk, phiG, phiK, theta, sample, N, cut, k_pts, weight_exponent = fs3.get_pars(0)
print("-----------PARAMETRS CHOSEN-----------")
print("Monolayers' tight-binding parameters: ",monolayer_type)
print("Symmetry of interlayer coupling: ",interlayer_symmetry," with values from sample ",sample)
print("Moiré potential values (eV,deg): G->(?,"+"{:.1f}".format(phiG/np.pi*180)+"°), K->("
      +"{:.4f}".format(Vk)+","+"{:.1f}".format(phiK/np.pi*180)+"°)")
print("Twist angle: "+"{:.2f}".format(theta)+"° and moiré length: "+"{:.4f}".format(cfs.moire_length(theta/180*np.pi))+" A")
print("Number of mini-BZs circles: ",N)
print("Computing over BZ cut: ",cut," with ",k_pts," points")
#Monolayer parameters
pars_monolayer = fs3.import_monolayer_parameters(monolayer_type,machine)
#Interlayer parameters
pars_interlayer = [interlayer_symmetry,np.load(fs3.get_pars_interlayer_fn(sample,interlayer_symmetry,monolayer_type,machine))]
distances = np.zeros(len(list_Vg))
for iVg in range(len(list_Vg)):
    Vg = list_Vg[iVg]
    print("Vg = ","{:.4f}".format(Vg))
    #Moire parameters
    pars_moire = fs3.import_moire_parameters(N,(Vg,Vk,phiG,phiK),theta)
    look_up = fs3.lu_table(pars_moire[0])
    K_point = np.array([0,0])       #Gamam
    energies = np.zeros(pars_moire[1]*44)
    weights = np.zeros(pars_moire[1]*44)
    H_tot = fs3.big_H(K_point,look_up,pars_monolayer,pars_interlayer,pars_moire)
    energies,evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
    ab = np.absolute(evecs)**2
    ind_MB = 22 #index of main band of the layer
    weights = np.sum(ab[:ind_MB,:],axis=0) + np.sum(ab[ind_MB*pars_moire[1]:ind_MB*pars_moire[1]+ind_MB,:],axis=0)
    inds = np.argsort(weights[pars_moire[1]*26:pars_moire[1]*28])
    ee = energies[pars_moire[1]*26+inds]
    distances[iVg] = ee[-2]-ee[-3]

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(list_Vg,distances,'b*-')
ax.axhline(y=0.09,color='r')
ax.set_xlabel(r"$V_\Gamma$")
ax.set_ylabel(r"Energy distance at $\Gamma$ between main band and $X$")
plt.show()

