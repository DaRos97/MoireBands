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
import utils
from pathlib import Path
import matplotlib.pyplot as plt
machine = cfs.get_machine(os.getcwd())

data_dn = Path(utils.get_home_dn(machine) + "Data/")

pars = []
s_a = []
c2 = []
for folder in data_dn.glob("temp*"):
    if folder.is_dir():
        # Extract spec_args
        name = folder.name
        tokens = name.split('_')
        args_minimization = {
            'TMD': tokens[1],
            'pts': int(tokens[2]),
            'Ks': tuple([float(tokens[3+i]) for i in range(6)]),
            'Bs': tuple([float(tokens[9+i]) for i in range(4)]),
        }
        # Tight-binding parameters
        npy_file = next(folder.glob("*.npy"), None)
        if not npy_file is None:
            s_a.append(args_minimization)
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
    args_minimization = s_a_sorted[ip]
    if not (          #Chose specific set: 1
        args_minimization['TMD']=="WS2" and     #TMD
        #args_minimization['Ks'][0]==1e-2 and     #K1
        #args_minimization['Ks'][1]>=1e-2 and     #K2
        #args_minimization['Ks'][3]==1 and     #K3
        #args_minimization['Ks'][3]==1 and     #K4
        #args_minimization['Ks'][4]==5 and     #K5
        #1e-6,1e-3,1,0.1,1
        #1e-6,1e-3,1,0.1,5
        #1e-6,1e-4,1,1,5
        1
    ):
        continue
    print(args_minimization)
    full_pars = pars_sorted[ip]
    chi2 = c2_sorted[ip]
    # Import data
    data = cfs.monolayerData(args_minimization['TMD'],master_folder,pts=args_minimization['pts'])
    # Plot
    HSO = cfs.find_HSO(full_pars[-2:])
    best_en = cfs.energy(full_pars,HSO,data.fit_data,args_minimization['TMD'])
    utils.plotResults(
        full_pars,best_en,data.fit_data,args_minimization,machine,chi2,
        show=True,
        #which=['orb',]#band']
    )

    continue
    if input("Show next best? [Y/n]")=='n':
        exit()
