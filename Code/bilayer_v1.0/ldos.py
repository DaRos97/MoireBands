""" LDOS in real space for selected parameters.
Changing centerPoint between 'K' and 'G' defines where we are computing the LDOS.

"""
import sys,os
import numpy as np
import scipy
cwd = os.getcwd()
master_folder = cwd[:40]
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_moire as fsm
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

sample = 'S11'
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees
disp = True
save_E = True
save_l = True
centerPoint = 'K'

""" Precision """
kPts = 20       # grid kPts x kPts in evals     # 30
rPts = 100          # 100
eMin, eMax, ePts = (-0.80,-0.6,200) if centerPoint=='G' else (-0.7,-0.4,200)     #-0.8 to -0.65, 200 usually
kFlat = fsm.LDOS_kList(kPts,theta,center=centerPoint)
eList = np.linspace(eMin,eMax,ePts)
spreadE = 0.01
rList = fsm.LDOS_rList(rPts,theta)

#eMin_G, eMax_G, ePts_G = (-0.80,-0.6,200)     #-0.8 to -0.65, 200 usually
#eMin_K, eMax_K, ePts_K = (-0.7,-0.4,300)     #-0.8 to -0.65, 200 usually

""" Fixed parameters """
nShells = 2
monolayer_type = 'fit'
nCells = int(1+3*nShells*(nShells+1))

""" Interlayer coupling """
Vg = 0.017              # eV
phiG = 170/180*np.pi        # rad
Vk = 0.006              # eV
phiK = 120/180*np.pi       # rad
w1p = -1.66         # eV
w1d = 0.324         # eV

stacking = 'P'
w2p=w2d = 0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}

if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Interlayer coupling w1: %f, %f"%(w1p,w1d))
    print("Sample ",sample," which has twist ",theta,"°")
    print("Moiré potential at G (%f eV, %f°)"%(Vg,phiG/np.pi*180))
    print("Moiré potential at K (%f eV, %f°)"%(Vk,phiK/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)
    print("(stacking,w2_p,w2_d,phi) = (%s, %.4f eV, %.4f eV, %.1f°)"%(stacking,w2p,w2d,phiG/np.pi*180))

args_e = (sample,centerPoint,nShells,monolayer_type,Vg,phiG,Vk,phiK,theta,stacking,w1p,w1d,phiG,kPts)
args_l = args_e + (rPts,eMin,eMax,ePts,spreadE)
l_fn = cfs.getFilename(('ldos_l',)+args_l,dirname='Data/',extension='.npy')
if not Path(l_fn).is_file():
    e_fn = cfs.getFilename(('ldos_e',)+args_e,dirname='Data/',extension='.npz')
    if not Path(e_fn).is_file():
        moire_pars = (Vg,Vk,phiG,phiK)
        args_e = (nShells, nCells, kFlat, monolayer_type, parsInterlayer, theta, moire_pars, '', False, True)
        evals, evecs = fsm.diagonalize_matrix(*args_e)
        if save_E:
            np.savez(e_fn,evals=evals,evecs=evecs)
    else:
        evals = np.load(e_fn)['evals']
        evecs = np.load(e_fn)['evecs']
    args_LDOS = (rList, eList, nShells, nCells, theta, kFlat, spreadE)
    LDOS = fsm.compute_LDOS(evals,evecs,*args_LDOS)
    if save_l:
        np.save(l_fn,LDOS)
else:
    LDOS = np.load(l_fn)

fig = plt.figure(figsize=(8,5))
s_ = 20
ax = fig.add_subplot(111)
X,Y = np.meshgrid(eList,np.linalg.norm(rList,axis=1))
rL = np.linalg.norm(rList[-1])
mesh = ax.pcolormesh(
    X,Y,
    LDOS[::-1,:],
    vmin=LDOS.min(),
    vmax=LDOS.max(),
    cmap='hot'
)
cbar = fig.colorbar(
    mesh,
    ax=ax,
)
ax.set_yticks([0,rL//3,2*rL//3,rL],
               [r"W/W",r"W/S",r"Se/W",r"W/W",],
               size=s_)     #Changed order because we're plotting the y-axis inverted
ax.set_xlabel('Energy [eV]',size=s_)

plt.show()











