"""
Here we do a simplified plot without stacking, w2p and w2d
"""
import sys,os
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
import functions_moire as fsm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
machine = cfs.get_machine(os.getcwd())

if len(sys.argv) != 2:
    print("Usage: python3 plot_edc.py arg1")
    print("arg1: nShells (1,2,3..)")
    quit()
else:
    nShells = int(sys.argv[1])

data = {}
for sample in ['S3','S11']:
    data_fn = 'Data/EDC/Vbest_'+fsm.get_fn(*(sample,nShells))+'.svg'
    if Path(data_fn).is_file():
        data[sample] = []
        with open(data_fn,'r') as f:
            l = f.readlines()
            for i in l:
                terms = i.split(',')
                data[sample].append([float(terms[3])/np.pi*180,float(terms[4])])
    else:
        print("Data file not found: ",data_fn)

data['S3'] = np.array(data['S3'])
data['S11'] = np.array(data['S11'])

marker = {'S3':'o', 'S11':'^'}
colors = {'S3':'r', 'S11':'b'}
label_phi = r'$\varphi$ °'

fig = plt.figure(figsize=(10,10))
s_ = 20
ax = fig.add_subplot()
for i_s in range(2):
    sample = 'S3' if i_s==0 else 'S11'
    ax.scatter(
        data[sample][:,0],             #x
        data[sample][:,1],               #y
        marker=marker[sample],            #marker
        lw=0,
        c=colors[sample],
        alpha=1,
        label=sample
    )
ax.set_xlabel(label_phi,size=s_)
ax.set_ylabel("Best V (eV)",size=s_)
ax.legend(
          loc='upper center',
          fontsize=s_,
         )


plt.show()
if input("Save fig? [y/N]")=='y':
    fig.savefig("Figures/edc_"+str(nShells)+'.png')
