""" Here we show the raw data and how it is processed to be used in the fitting.
"""
import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':        #local machine
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':    #HPC
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':           #Mafalda
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_monolayer as fsm
from pathlib import Path
import matplotlib.pyplot as plt

TMD = 'WSe2'
ptsPerPath = (20,20) if TMD=="WS2" else (40,20,20)

data = cfs.dataWS2() if TMD=='WS2' else cfs.dataWSe2()
paths = data.paths
raw_data = data.raw_data

if 0:   #Plot raw
    fig,axs = plt.subplots(1,2,figsize=(20,10))
    for ip,path in enumerate(paths):
        ax = axs[ip]
        nbands = data.nbands[path]
        for ib in range(nbands):
            ax.plot(
                raw_data[path][ib][:,0],
                raw_data[path][ib][:,1],
                marker='o',
                lw=0.5,
                label="Band %d"%(ib+1),
            )
        ax.set_title("path "+path,size=20)

    ax.legend()
    plt.suptitle("Raw data",size=30)
    fig.tight_layout()

""" Symmetrize """
sym_data = data.sym_data

if 0:   #Plot symm
    fig,axs = plt.subplots(1,2,figsize=(20,10))
    for ip,path in enumerate(paths):
        ax = axs[ip]
        nbands = data.nbands[path]
        for ib in range(nbands):
            ax.plot(
                sym_data[path][ib][:,0],
                sym_data[path][ib][:,1],
                marker='o',
                lw=0.5,
                label="Band %d"%(ib+1),
            )
        ax.set_title("path "+path,size=20)

    plt.suptitle("Symmetrized data",size=30)
    fig.tight_layout()

""" Merge """
mer_data = data.getFitData(ptsPerPath)

if 1:   #Plot mer
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()
    nbands = 6 if TMD=='WSe2' else 2
    for ib in range(nbands):
        ax.plot(
            mer_data[:,0],
            mer_data[:,3+ib],
            marker='o',
            lw=0.5,
            label="Band %d"%(ib+1),
        )

    plt.suptitle("Merged and reduced data "+TMD,size=30)
    plt.legend()
    fig.tight_layout()



plt.show()












