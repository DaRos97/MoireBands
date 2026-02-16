""" Here I plot the EDC and maybe also the full weight image around Gamma of the solutions.
"""
import numpy as np
import os,sys
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
machine = cfs.get_machine(cwd)

save = False
sample = "S11"
nShells = 2
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees, from LEED eperiment
home_dn = fsm.get_home_dn(machine)
data_dn = cfs.getFilename(('edc',*(sample,nShells,theta)),dirname=home_dn+"Data/newEDC/")+'/'
full_fn = data_dn + "full.npy"

if Path(full_fn).is_file():
    data = np.load(full_fn)     #ns,4 -> each solution has w1p,w1d,phase(rad) and bestV
else:
    raise ValueError("Not found full.npy file.")

nCells = int(1+3*nShells*(nShells+1))
kListG = np.array([np.zeros(2),])
spreadE = 0.03      # in eV
monolayer_type = 'fit'
w2p = w2d = 0
stacking = 'P'
Vk,phiK = (0.007,-106/180*np.pi)

ns = data.shape[0]
phiMin = 172/180*np.pi
phiMax = 177/180*np.pi
Vmin = 15/1000
Vmax = 16.2/1000
mask = (data[:,2]>phiMin) & (data[:,2]<phiMax) & (data[:,3]>Vmin) & (data[:,3]<Vmax)
nsRed = data[mask].shape[0]
print(nsRed," solutions")

for i in range(ns):
    w1p,w1d,phiG,Vg = data[i]
    if phiG<phiMin or phiG>phiMax or Vg<Vmin or Vg>Vmax:
        continue
    print(w1p,w1d,phiG,Vg)
    figname = cfs.getFilename(*('edc',sample,theta,w1p,w1d,phiG,Vg),dirname="Figures/newEDC/",extension='.png')
    parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
    args = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
    fsm.EDC(
        args,sample,spreadE=spreadE,
        disp=False,
        plot=True,
        #figname=figname,
        show=True
    )







