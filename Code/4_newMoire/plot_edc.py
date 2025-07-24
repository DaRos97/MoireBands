"""
Here we plot all best values of V for many parameter choises.
We have 3 parameters: w1p,w1d and phi.
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
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
machine = cfs.get_machine(os.getcwd())

nShells = 2

fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot()
for sample in ['S3','S11']:
    data_fn = 'Data/EDC/Vbest_'+fsm.get_fn(*(sample,nShells))+'.svg'
    if Path(data_fn).is_file():
        data = {'P':[],'AP':[]}
        with open(data_fn,'r') as f:
            l = f.readlines()
            for i in l:
                terms = i.split(',')
                if terms[0]!='P':
                    continue
                data[terms[0]].append([float(terms[1]),float(terms[2]),float(terms[3])/np.pi*180,float(terms[4])])
    else:
        print("Data file not found")
        quit()

    data['P'] = np.array(data['P'])
    data['AP'] = np.array(data['AP'])

    # Choice of parameters: 0->w1p, 1->w1d, 2->phi
    par1 = 2
    par2 = 1
    par3 = 0
    marker = {'P':'o', 'AP':'^'} if sample == 'S3' else {'P':'*', 'AP':'^'}
    colormap = {'P':'coolwarm','AP':'coolwarm'} if sample == 'S3' else {'P':'inferno_r','AP':'viridis'}
    labels = [r'$w_1^p$ [eV]',r'$w_1^d$ [eV]',r'$\varphi$ °']

    s_ = 20
    stacking = 'P'
    # Sizes -> par2
    sizes = np.copy(data[stacking][:,par2])
    if np.max(abs(sizes))>0:
        sizes /= np.max(abs(sizes))     #Normalize
    sizes = sizes**10     #make it larger
    sizes = sizes*25
    #sizes -= np.min(sizes) - 5     #Bring from 0 to 1 and add offset
    # Colors -> par3
    colorValues = data[stacking][:,par3]
    norm = mcolors.Normalize(vmin=np.min(colorValues), vmax=np.max(colorValues))
    cmap = matplotlib.colormaps[colormap[stacking]]
    colors = cmap(norm(colorValues))  # Array of RGBA tuples
    ax.scatter(
        data[stacking][:,par1],             #x
        data[stacking][:,-1],               #y
        marker=marker[stacking],            #marker
        lw=0,
        s=sizes**2,
        c=colors,
        alpha=0.8
    )
    # Colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location='right' if sample=='S3' else 'left')
    cbar.set_label(labels[par3],size=s_)
    #cax.yaxis.set_label_position('left')
    #cax.yaxis.set_ticks_position('left')

    # Legend
    if sample == 'S11':
        ax = ax.twinx()
        ax.axis('off')
    else:
        ax.set_xlabel(labels[par1],size=s_)
    sizes1 = np.unique(data[stacking][:,par2])
    sizes2 = np.unique(sizes)
    legend_elements = [ Line2D([0], [0],
                       marker=marker[stacking], color='gray',
                       label="{:.3f}".format(label),
                       markerfacecolor='gray',
                       markeredgecolor='k',
                       markersize=size,
                       linewidth=0,
                       markeredgewidth=0.5
                      )     for size, label in zip(sizes2, sizes1)
    ]
    ax.legend(handles=legend_elements,
              loc='lower center' if sample == 'S3' else 'upper center',
              title=labels[par2],
              fontsize=s_-10,
              title_fontsize=s_
             )

ax.set_title("Best V [eV]",size=s_)

#ax.set_title(stacking,size=s_)

#plt.suptitle("Sample="+sample+", nShells=%d"%nShells,size=s_)
fig.tight_layout()
plt.show()
if input("Save fig? [y/N]")=='y':
    fig.savefig("Figures/edc_"+str(nShells)+'.png')
