"""
Here we take an intensity cut at Gamma to look at the distance between the main band and the side band crossing below it.
For the 4-dimensional space of V, phi, w1p and w1d we define a distance from the ARPES result in two ways:
    - Considering distances: d1,d2 distance of side band crossing(sbc) and of WS2 band from TVB, respectively
    - Considering positions: p1,p2,p3 positions of TVB, sbc and WS2, respectively.
From the ARPES exp we have, for S11 [eV]:
    - d1=0.0932, d2=0.6601
    - p1=-1.1599, p2=-1.2531, p3=-1.8200
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
import utils
import pandas as pd
from pathlib import Path

machine = cfs.get_machine(os.getcwd())
n_chunks = 128

""" Parameters and options """
if len(sys.argv)!=2:
    print("Usage: python3 edc.py arg1")
    print("arg1 is an int for the index of the parameter list")
    exit()
sample = 'S11'
ind = int(sys.argv[1])
if machine=='maf':
    ind -= 1
disp = machine=='loc'

""" Fixed parameetrs """
theta_deviation = 0      #Change here for \pm 0.3 degrees
nShells = 2
Vk,phiK = (0.006,106/180*np.pi)
spreadE = 0.03      # in eV
#
nCells = cfs.get_nCells(nShells)
monolayer_type = 'fit'
theta = cfs.dic_params_twist[sample]+theta_deviation    #twist angle, in degrees, from LEED eperiment
kListG = np.array([np.zeros(2),])
stacking = 'P'
w2p = w2d = 0
if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Sample ",sample," with twist %.2f°"%theta," (variation of %.1f° from LEED)"%theta_deviation)
    print("Moiré potential at K (%.5f eV, %.1f°)"%(Vk,phiK/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)
    print("Energy spreading: %.3f eV"%spreadE)

""" Computation """
parameters_chunk, listFn = utils.get_parameters(ind,n_chunks=n_chunks)
results = []
for Vg,phiG,w1p,w1d in parameters_chunk:
    parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
    args_diag = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
    positions,success = utils.EDC(
        args_diag,
        sample,
        spreadE=spreadE,
        disp=False,
        plot=False,
        machine=machine
    )
    if success:
        results.append((Vg,phiG,w1p,w1d,*positions))
    else:
        results.append((Vg,phiG,w1p,w1d,np.nan,np.nan,np.nan))
    if disp:
        print("Vg: %.3f\tphiG: %.1f\tw1p: %.3f\t w1d: %.3f"%(Vg,phiG/np.pi*180,w1p,w1d))
        #print("Iteration time: %.4f"%(tf-ti))

""" Save to file: hdf5 """
df = pd.DataFrame(
    results,
    columns=["Vg", "phiG", "w1p", "w1d", "p1", "p2", "p3"]
)
dirname = cfs.getFilename(
    ('edcGamma',theta_deviation,nShells,Vk,phiK,spreadE),
    dirname=utils.get_home_dn(machine)+"Data/",
    floatPrecision=3,
) + listFn + '/'
if not Path(dirname).is_dir():
    os.system("mkdir "+dirname)
output_file = dirname + "chunk_%d_%d.h5"%(ind,n_chunks)
df.to_hdf(
    output_file,
    key="results",
    mode="w",
    format="table",      # allows later append/select
    complevel=5,
    complib="blosc"
)

print("Finished chunk %d/%d"%(ind,n_chunks))




















