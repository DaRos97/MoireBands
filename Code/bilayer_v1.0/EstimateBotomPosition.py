""" Here I simply import the sample experiemntal image and extract the peak of the bottom band.
"""

import sys,os
import argparse
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
from tqdm import tqdm
machine = cfs.get_machine(os.getcwd())
from PIL import Image

sample = "S11"
E_max, E_min, pKi, pKf, pEmax, pEmin = cfs.dic_pars_samples[sample]
fig_fn = fsm.get_inputs_dn(machine) + sample + '_KGK.png'
pic_raw = np.array(np.asarray(Image.open(fig_fn)))
totPe,totPk,_ = pic_raw.shape
kList = cfs.get_kList('Kp-G-K',11)
K0 = np.linalg.norm(kList[0])   #val of |K|
pK0 = int((pKf+pKi)/2)   #pixel of middle -> k=0
pKF = int((pKf-pK0)*K0+pK0)   #pixel of k=|K|
pKI = 2*pK0-pKF             #pixel of k=-|K|
exp_pic = pic_raw[pEmax:pEmin,pKI:pKF]


fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(131)
ax.imshow(exp_pic)

ax = fig.add_subplot(132)
indMid = exp_pic.shape[1] // 2
nE = exp_pic.shape[0]
enList = np.linspace(-3.5,0,nE)
intensities = np.array(255 - exp_pic[:,indMid,0][::-1],dtype=np.float64)
print(intensities)
ax.plot(
    enList,
    intensities
)


# Fit 
minE = -1.9
maxE = -1.65
ax.axvline(minE,c='r',ls='--')
ax.axvline(maxE,c='r',ls='--')

enList = np.linspace(minE,maxE,200)
minI = int(minE*nE/3.5)
maxI = int(maxE*nE/3.5)
ints = intensities[minI:maxI]
enListPlot = np.linspace(minE,maxE,ints.shape[0])

ax = fig.add_subplot(133)
ax.plot(
    enListPlot,
    ints
)

import lmfit
model = lmfit.Model(fsm.voigt)#, independent_vars=['x'])
params = model.make_params(
    amplitude=1.57,
    center=-1.8,
    gamma=0.03,
    sigma=0.07,
)
params['sigma'].set(min=1e-6, max=50)        # Gaussian width
params['gamma'].set(min=1e-6, max=50)       # Lorentzian widths
params['amplitude'].set(min=0)
result = model.fit(ints, params, x=enListPlot)

ax.plot(
    enList,
    result.eval(x=enList)
)
ax.axvline(result.best_values['center'],c='r',ls='--')
ax.set_title(r"Center: %.4f $\pm$ %.5f eV"%(result.best_values['center'],0.0025))
print(result.fit_report())

plt.show()
