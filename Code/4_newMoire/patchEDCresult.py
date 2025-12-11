""" Here we go through all the single files and put them together since there should not be many results anyway.
"""
import numpy as np
import os
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

save = True
sample = "S11"
nShells = 2
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees, from LEED eperiment
listPhi = np.linspace(0,2*np.pi,72,endpoint=False)
home_dn = fsm.get_home_dn(machine)
data_dn = cfs.getFilename(('edc',*(sample,nShells,theta)),dirname=home_dn+"Data/newEDC/")+'/'
full_fn = data_dn + "full.npy"

if Path(full_fn).is_file():
    full_data = np.load(full_fn)
else:
    folder = Path(data_dn)
    full_data = []
    for f in folder.glob(".npy"):
        if f.is_file():
            data = np.load(f)
            name = f.name
            w1p = float(name.split('_')[1])
            w1d = float(name.split('_')[2])
            for i in range(data.shape[0]):
                full_data.append([w1p,w1d,listPhi[int(data[i,0])],data[i,1]])

    if save:
        np.save(full_fn,np.array(full_data))












