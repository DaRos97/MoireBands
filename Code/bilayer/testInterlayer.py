""" Here I test if the EDC depends on the interlayer coupling or just on V,phi.
"""
import sys,os
import numpy as np
import scipy
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_moire as fsm
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools
import csv
machine = cfs.get_machine(os.getcwd())

fit_disp = False
fit_plot = False
sample = 'S11'
peak0 = -0.6948 if sample=='S3' else -0.6899
peak1 = -0.7730 if sample=='S3' else -0.7831
peaks = (peak0,peak1)
nShells = 2
monolayer_type = 'fit'
Vk,phiK = (0.007,-106/180*np.pi)
nCells = int(1+3*nShells*(nShells+1))
theta = 2.8    #twist angle, in degrees, from LEED eperiment
kListG = np.array([np.zeros(2),])
spreadE = 0.03      # in eV
Vg = 0.017
phiG = 175/180*np.pi
stacking = 'P'
w2p = w2d = 0

# Grid of interlayer coupling
Nwp = 11
Nwd = 8
listW1p = np.linspace(-1.620,-1.680,Nwp)         # every 2 meV
listW1d = np.linspace(0.310,0.340,Nwd)           # every 2 meV

fn = "Data/testInt_%.3f_%s_%d_%.3f_%.3f_%d_%.3f_%.3f.npy"%(Vg,int(phiG/np.pi*180),Nwp,listW1p[0],listW1p[-1],Nwd,listW1d[0],listW1d[-1],)
if Path(fn).is_file():
    EDC = np.load(fn)
else:
    EDC = np.zeros((Nwp,Nwd))
    for iwp in range(Nwp):
        print(iwp)
        for iwd in range(Nwd):
            w1p = listW1p[iwp]
            w1d = listW1d[iwd]
            parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
            args = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
            a = fsm.EDC(args,sample,peaks,spreadE=spreadE,disp=fit_disp,plot=fit_plot,figname='',testInterlayer=True)

            EDC[iwp,iwd] = abs(a[0]-a[1])
    np.save(fn,EDC)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
pm = ax.pcolormesh(
    listW1p,
    listW1d,
    EDC.T,
    cmap='plasma'
)
s = 25
ax.set_xlabel(r"$w_1^p$",size=s)
ax.set_ylabel(r"$w_1^d$",size=s)
cbar = fig.colorbar(pm)
cbar.set_label("EDC",size=s)
ax.set_title(r"V=%d meV, $\phi_G$=%d°"%(Vg*1e3,int(phiG/np.pi*180)),size=s)
plt.show()































