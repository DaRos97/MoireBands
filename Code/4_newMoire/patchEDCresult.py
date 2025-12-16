""" Here we go through all the single files and put them together since there should not be many results anyway.
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
listPhi = np.linspace(0,2*np.pi,72,endpoint=False)
home_dn = fsm.get_home_dn(machine)
data_dn = cfs.getFilename(('edc',*(sample,nShells,theta)),dirname=home_dn+"Data/newEDC/")+'/'
full_fn = data_dn + "full.npy"
checkBottomBand = True

if Path(full_fn).is_file():
    full_data = np.load(full_fn)
else:
    if checkBottomBand:
        nCells = int(1+3*nShells*(nShells+1))
        kListG = np.array([np.zeros(2),])
        spreadE = 0.03      # in eV
        monolayer_type = 'fit'
        w2p = w2d = 0
        stacking = 'P'
        Vk,phiK = (0.007,-106/180*np.pi)
    folder = Path(data_dn)
    full_data = []
    for f in folder.glob("*.npy"):
        if f.is_file():
            data = np.load(f)
            name = f.name
            w1p = float(name.split('_')[1])
            w1d = float(name.split('_')[2])
            if checkBottomBand:
                parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
            nSolutions = data.shape[0]
            for i in range(nSolutions):
                phiG,Vg = listPhi[int(data[i,0])], data[i,1]
                if checkBottomBand:
                    args = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
                    a = fsm.EDC(
                        args,sample,spreadE=spreadE,
                        disp=False,
                        plot=False,
                        #figname=figname,
                        show=False
                    )
                    if a[0]==0 and a[1]==0:
                        goodBottomBand = False
                    else:
                        goodBottomBand = True
                if not checkBottomBand or (checkBottomBand and goodBottomBand):
                    full_data.append([w1p,w1d,phiG,Vg])

    full_data = np.array(full_data)
    if save:
        np.save(full_fn,full_data)

if machine=='maf':
    exit()

""" Quick plot of solutions """
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

cmap = plt.get_cmap('plasma_r')
wpmin = np.min(full_data[:,0])
wpmax = np.max(full_data[:,0])
norm = Normalize(vmin=wpmin, vmax=wpmax)

wdmin = np.min(full_data[:,1])
wdmax = np.max(full_data[:,1])
sizes = 30+300*(full_data[:,1]-wdmin)/(wdmax-wdmin)

fig = plt.figure(figsize=(20,10))

ns = full_data.shape[0]     #Number of solutions
print("%d solutions"%ns)

phiMin = 172/180*np.pi
phiMax = 177/180*np.pi

s_ = 20
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
for i in range(ns):
    if full_data[i,2] < phiMin or full_data[i,2] > phiMax:
        continue
    ax1.scatter(
        full_data[i,2]/np.pi*180,
        full_data[i,3],
        color=cmap(norm(full_data[i,0])),
        marker='^',
        s=sizes[i],
        alpha=0.5,
        lw=0
    )

    ax2.scatter(
        full_data[i,0],
        full_data[i,3],
        color=cmap(norm(full_data[i,0])),
        marker='^',
        s=sizes[i],
        alpha=0.5,
        lw=0
    )

    ax3.scatter(
        full_data[i,1],
        full_data[i,3],
        color=cmap(norm(full_data[i,0])),
        marker='^',
        s=sizes[i],
        alpha=0.5,
        lw=0
    )

ax1.set_xlim(phiMin/np.pi*180,phiMax/np.pi*180)
ax1.set_ylabel("V moiré",size=s_)
ax1.set_xlabel("phase (°)",size=s_)
ax2.set_xlabel(r"$w_1^p$",size=s_)
ax3.set_xlabel(r"$w_1^d$",size=s_)

sm = ScalarMappable(norm=norm,cmap=cmap)
cax = fig.add_subplot([0.93,0.12,0.02,0.78])
plt.colorbar(sm,cax=cax)
plt.show()








