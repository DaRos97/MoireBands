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
from pathlib import Path
import matplotlib.pyplot as plt

""" Input processing """
inp = sys.argv
if len(inp)!=2:
    print("Usage: python visualize_ARPES_data.py arg\nwith arg a string containing only s, r and/or m for (r)aw, (s)ymmetrized and (f)it data.")
    exit()
for i in inp[1]:
    if i not in ['r','s','f']:
        raise ValueError("Input must contain only r, s and/or f.")

TMD = 'WS2'
pts = 91

data = cfs.monolayerData(TMD)
paths = data.paths

""" Raw data """
if 'r' in inp[1]:
    raw_data = data.raw_data
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
    plt.suptitle("Raw data %s"%TMD,size=30)
    fig.tight_layout()

""" Symmetrize """
if 's' in inp[1]:
    sym_data = data.sym_data
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

""" data for Fit """
if 'f' in inp[1]:
    fit_data = data.fit_data
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()
    nbands = 6 #if TMD=='WSe2' else 6
    for ib in range(nbands):
        ax.plot(
            fit_data[:,0],
            fit_data[:,3+ib],
            marker='o',
            lw=0.5,
            label="Band %d"%(ib+1),
        )

    plt.suptitle("Merged and reduced data "+TMD,size=30)
    plt.legend()
    fig.tight_layout()



plt.show()












