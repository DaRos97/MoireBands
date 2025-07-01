"""
Here we take a cut at Gamma to look at the distance between the main band and the side band crossing below it.
We compute this distance for a class of parameters:
    - V and psi at Gamma
    - w2p and w2d
    - theta
We compare this distance to the experimentally extracted one:
    - 78 meV for S3
    - 93 meV for S11
We then look also at the full image to see if the parameters make sense.

An extension would be to do the same at K
"""
import sys,os
import numpy as np
import scipy
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
machine = cfs.get_machine(os.getcwd())

if len(sys.argv)!=2:
    print("Usage: python3 EDC.py arg1")
    print("arg1: sample (S3 or S11)")
    quit()
else:
    sample = sys.argv[1]

# Preamble
disp = True

#Parameters
monolayer_type = 'fit'
stacking = 'P'
w2p,w2d = (0,0)
Vg,phiG = (0.02,np.pi)
Vk,phiK = (0,0)
nShells = 2
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees
w1p = cfs.w1p_dic[monolayer_type][sample]
w1d = cfs.w1d_dic[monolayer_type][sample]
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}

if disp:    #print what parameters we're using
    print("-----------PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Interlayer coupling: ",parsInterlayer)
    print("Sample ",sample," which has twist ",theta,"° and moiré length: "+"{:.4f}".format(cfs.moire_length(theta/180*np.pi))+" A")
    print("Moiré potential values (eV,deg): G->("+"{:.4f}".format(Vg)+","+"{:.1f}".format(phiG/np.pi*180)+"°), K->("
          +"{:.4f}".format(Vk)+","+"{:.1f}".format(phiK/np.pi*180)+"°)")
    print("Number of mini-BZs circles: ",nShells)

###################################################################################
# Diagonalize Hamiltonian to get eigenvalues and eigenvectors
###################################################################################
nCells = int(1+3*nShells*(nShells+1))
kList = np.array([np.zeros(2),])
args = (nShells, nCells, kList, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False)
evals, evecs = fsm.diagonalize_matrix(*args)

en_MB = evals[0][28*nCells]

ab = np.absolute(evecs[0])**2
weights = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)

w_MB = weights[24*nCells:28*nCells]

print(w_MB)





