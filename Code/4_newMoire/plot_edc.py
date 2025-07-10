"""
Here we plot all best values of V for many parameter choises.
We have 3 parameters: w2p,w2d and phi PLUS stacking: P or AP
We plot it in a 2D plot, with best V on y axis.
Parameter 1 goes on x axis
Parameter 2 is the size
Parameter 3 is the color
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
import matplotlib.colors as mcolors
machine = cfs.get_machine(os.getcwd())

if len(sys.argv) != 3:
    print("Usage: python3 plot_edc.py arg1 arg2")
    print("arg1: S3 or S11")
    print("arg2: nShells (1,2,3..)")
    quit()
else:
    sample = sys.argv[1]
    nShells = int(sys.argv[2])

data_fn = 'Data/EDC/Vbest_'+fsm.get_fn(*(sample,nShells))+'.svg'
if Path(data_fn).is_file():
    data = {'P':[],'AP':[]}
    with open(data_fn,'r') as f:
        l = f.readlines()
        for i in l:
            terms = i.split(',')
            data[terms[0]].append([float(terms[1]),float(terms[2]),float(terms[3]),float(terms[4])])
else:
    print("Data file not found")
    quit()

data['P'] = np.array(data['P'])
data['AP'] = np.array(data['AP'])

# Choice of parameters: 0->w2p, 1->w2d, 2->phi
par1 = 2
par2 = 0
par3 = 1
marker = {'P':'+', 'AP':'^'}
colormap = {'P':'plasma','AP':'viridis'}
labels = [r'$w_2^p$',r'$w_2^d$',r'$\varphi$']

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot()
sm = {}
for stacking in ['P','AP']:
    # Sizes -> par2
    sizes = data[stacking][:,par2]
    sizes /= np.max(abs(sizes)) * 50
    # Colors -> par3
    colorValues = data[stacking][:,par3]
    norm = mcolors.Normalize(vmin=np.min(colorValues), vmax=np.max(colorValues))
    cmap = cm.get_cmap(colormap[stacking])
    colors = cmap(norm(colorValues))  # Array of RGBA tuples
    ax.scatter(
        data[stacking][:,par1],             #x
        data[stacking][:,-1],               #y
        marker=marker[stacking],            #marker
        s=sizes,
        c=colors,
        label=stacking
    )
    sm_ = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_.set_array([])  # Required for colorbar, can be empty
    sm[stacking] = sm_

#Colorbars
cax1 = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # right of plot
cax2 = fig.add_axes([0.96, 0.3, 0.02, 0.4])  # right of the first colorbar
cbar1 = fig.colorbar(sm['P'], cax=cax1)
cbar1.set_label(labels[par3])
cbar2 = fig.colorbar(sm['AP'], cax=cax2)
cbar2.set_label(labels[par3])

s_ = 20
ax.legend(size=s_)
ax.set_title("Sample="+sample+", nShells=%d"%nShells,size=s_)
ax.set_xlabel(xlabels[par1],size=s_)
ax.set_ylabel("Best V",size=s_)

plt.show()
