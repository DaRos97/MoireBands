""" Here I go through the many results plotting the in order of total chi2.
"""
import sys,os
import glob
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_monolayer as fsm
from pathlib import Path
import matplotlib.pyplot as plt
machine = cfs.get_machine(os.getcwd())

data_dn = Path(fsm.get_home_dn(machine) + "Data/")

pars = []
s_a = []
c2 = []
for folder in data_dn.glob("temp*"):
    if folder.is_dir():
        # Extract spec_args
        name = folder.name
        tokens = name.split('_')
        values = [tokens[1],]
        values += [
            float(tok) if '.' in tok else int(tok)
            for tok in tokens[2:-3]
        ]
        ptsPerPath = tuple([int(tok) for tok in tokens[-3:]])
        values.append(ptsPerPath)
        spec_args = tuple(values)
        # Tight-binding parameters
        npy_file = next(folder.glob("*.npy"), None)
        if not npy_file is None:
            s_a.append(spec_args)
            tb = np.load(npy_file)
            pars.append(tb)
            # Chi2
            chi2 = float(npy_file.name.split('_')[1][:-4])
            c2.append(chi2)

# Sort chi2 and corresponding parameters and spec_args
Npars =len(c2)
order = sorted(range(Npars), key=lambda i: c2[i])
c2_sorted = [c2[i] for i in order]
pars_sorted = [pars[i] for i in order]
s_a_sorted = [s_a[i] for i in order]

# Plot in order
for ip in range(Npars):
    spec_args = s_a_sorted[ip]
    full_pars = pars_sorted[ip]
    chi2 = c2_sorted[ip]
    # Import data
    TMD = spec_args[0]
    ptsPerPath = spec_args[-1]
    print(ptsPerPath)
    dataObject = cfs.dataWS2() if TMD=="WS2" else cfs.dataWSe2()
    data = dataObject.getFitData(ptsPerPath)
    # Plot
    HSO = cfs.find_HSO(full_pars[-2:])
    best_en = cfs.energy(full_pars,HSO,data,spec_args[0])
    fsm.plotResults(full_pars,best_en,data,spec_args,machine,chi2,show=True)

    if not input("Show next best? [y/N]")=='y':
        exit()
