import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import sys,os
cwd = os.getcwd()
master_folder = cwd[:43]
sys.path.insert(1, master_folder)
import CORE_functions as cfs
from pathlib import Path
import functions as fs
import pickle

save_data = 0#True

fig = plt.figure(figsize=(20,10))

#Parameters
side_bands = 5
momentum_points = 1001
V_list = np.linspace(0.015,0.02,1)
aM_list = np.linspace(10,25,1)
phi = 0
mass = 1
factor = -1.7       #place where to compute the side band distance
args_fn = (side_bands,momentum_points,V_list[0],V_list[-1],len(V_list),aM_list[0],aM_list[-1],len(aM_list),phi,mass,factor)
data_fn = fs.get_data_fn(*args_fn)

if not Path(data_fn).is_file():
    args = (side_bands,momentum_points,V_list,aM_list,phi,mass,factor)
    data = fs.compute_data(save_data,data_fn,*args)
else:
    with open(data_fn,'rb') as f:
        data = pickle.load(f)

################################################################
plot_1a = True
"""
Fig 2.a -> 1D bands and side bands
"""

if plot_1a:
    ax = fig.add_subplot(121)
    #
    c = ['k','g','r','r','y','m','c']
    LW = 0.1    #line width
    #Data
    iv,ia = (0,0)
    mrl = 2*np.pi/aM_list[ia]
    e_ = -(factor*mrl)**2/2/mass        #energy of main band between 2nd and third side band max
    #Bands and weights
    for t in range(2*side_bands+1):
        ax.plot(data['momenta'][iv,ia],data['energies'][iv,ia,t],color='k',lw=LW,zorder=-1)
        ax.scatter(data['momenta'][iv,ia],data['energies'][iv,ia,t],s=data['weights'][iv,ia,t]**(1/2)*100,c='b',lw=0)
    #Gap arrow
    ax.arrow(-mrl/2,data['energies'][iv,ia,-1][fs.ind_k(-mrl/2,data['momenta'][iv,ia])],
             0,-fs.gap(data['energies'][iv,ia],-mrl/2,data['momenta'][iv,ia]),
             color='r',
             label='Gap',
             arrowstyle='<->',
             head_length=0.5,
             lw=2,
#             width=0
            )
    #Horizontal distance arrow
    l, inds = fs.horizontal_displacement(e_,data['energies'][iv,ia],data['weights'][iv,ia],data['momenta'][iv,ia])
    i_mb,i_sb1h,i_sb2h = inds
    ax.scatter(data['momenta'][iv,ia][l[i_mb,1]],data['energies'][iv,ia][l[i_mb,0],l[i_mb,1]],c='k',s=150)
    ax.scatter(data['momenta'][iv,ia][l[i_sb1h,1]],data['energies'][iv,ia][l[i_sb1h,0],l[i_sb1h,1]],c='lime',s=100)
    ax.scatter(data['momenta'][iv,ia][l[i_sb2h,1]],data['energies'][iv,ia][l[i_sb2h,0],l[i_sb2h,1]],c='g',s=100)

    ax.arrow(data['momenta'][iv,ia][l[i_mb,1]],data['energies'][iv,ia][l[i_mb,0],l[i_mb,1]],
            data['momenta'][iv,ia][l[i_sb1h,1]]-data['momenta'][iv,ia][l[i_mb,1]],0,
            color='lime',label='inner band',head_length=0,width=0)
    ax.arrow(data['momenta'][iv,ia][l[i_mb,1]],data['energies'][iv,ia][l[i_mb,0],l[i_mb,1]],
            data['momenta'][iv,ia][l[i_sb2h,1]]-data['momenta'][iv,ia][l[i_mb,1]],0,
            color='g',label='outer band',head_length=0,width=0)
    #Vertical distance arrow
    i_k = l[i_mb,1]
    i_mb,i_sb1,i_sb2 = fs.vertical_displacement(i_k,data['weights'][iv,ia])
    ax.scatter(data['momenta'][iv,ia][i_k],data['energies'][iv,ia][i_mb,i_k],c='m',s=150,zorder=10)
    ax.scatter(data['momenta'][iv,ia][i_k],data['energies'][iv,ia][i_sb1,i_k],c='aqua',s=100)
    ax.scatter(data['momenta'][iv,ia][i_k],data['energies'][iv,ia][i_sb2,i_k],c='dodgerblue',s=100)

    e_lower = data['energies'][iv,ia][i_sb1,i_k]-data['energies'][iv,ia][i_mb,i_k]
    e_upper = data['energies'][iv,ia][i_sb2,i_k]-data['energies'][iv,ia][i_mb,i_k]
    ax.arrow(data['momenta'][iv,ia][i_k],data['energies'][iv,ia][i_mb,i_k],     #x,y,dx,dy
            0,e_lower,
            color='aqua',label='lower band',head_length=0,width=0)
    ax.arrow(data['momenta'][iv,ia][i_k],data['energies'][iv,ia][i_mb,i_k],
            0,e_upper,
            color='dodgerblue',label='upper band',head_length=0,width=0)

    #Plot features
    ax.legend(fontsize=20,loc='lower center')
    #ax.set_title(title)
    #Limits
    rg = np.max(data['energies'][iv,ia])-np.min(data['energies'][iv,ia])
    ax.set_ylim(data['energies'][iv,ia][i_sb1,i_k]-rg*0.01,np.max(data['energies'][iv,ia][-1])+rg*0.01)
    ax.set_xlim(data['momenta'][iv,ia][l[i_sb2h,1]]-mrl,-data['momenta'][iv,ia][l[i_sb2h,1]]+mrl)
    #X ticks and vertical lines
    ax.set_xticks([-2*mrl,-mrl,0,mrl,2*mrl],[r'$-2G$',r'$-G$',r'$\Gamma$',r'$G$',r'$2G$'],size=30)
    for i in range(5):
        ax.plot([-2*mrl+i*mrl,-2*mrl+i*mrl],[-10,10],color='k',lw=0.5,zorder=-1)
    ax.set_yticks([])
    ax.set_ylabel("Energy",size=30)

plt.tight_layout()

plt.show()

