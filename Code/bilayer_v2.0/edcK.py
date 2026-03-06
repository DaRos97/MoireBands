"""
Here we take an intensity cut at K to look at the distance between the main band and the side band crossing below it.
In this situation there is no role of interlayer coupling, so we just look at the distance between main band and moiré bands.
From the ARPES exp we have, for S11 [eV]:
    - d=0.170 eV
    - p=-0.8990 eV
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
    print("Usage: python3 edcK.py arg1")
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
Vg,phiG = (0.019,175/180*np.pi)
spreadE = 0.03      # in eV
w1p = -1.750
w1d = 1.050
#
nCells = cfs.get_nCells(nShells)
monolayer_type = 'fit'
theta = cfs.dic_params_twist[sample]+theta_deviation    #twist angle, in degrees, from LEED eperiment
kListK = np.array([[4*np.pi/3/cfs.dic_params_a_mono['WSe2'],0],])
stacking = 'P'
w2p = w2d = 0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Sample ",sample," with twist %.2f°"%theta," (variation of %.1f° from LEED)"%theta_deviation)
    print("Moiré potential at Gamma (%.5f eV, %.1f°)"%(Vg,phiG/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)
    print("Energy spreading: %.3f eV"%spreadE)

""" Computation """
parameters_chunk, listFn = utils.get_parametersK(ind,n_chunks=n_chunks)
results = []
for Vk,phiK in parameters_chunk:
    #Vk = 0.0
    #phiK = 120/180*np.pi
    args_diag = (nShells, nCells, kListK, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
    positions,success = utils.EDC(
        args_diag,
        sample,
        BZpoint='K',
        spreadE=spreadE,
        disp=False,
        plot=False,
        machine=machine
    )
    if success:
        results.append((Vk,phiK,*positions))
    else:
        results.append((Vk,phiK,np.nan,np.nan))
    if disp:
        print("Vk: %.3f\tphiK: %.1f"%(Vk,phiK/np.pi*180))

""" Save to file: hdf5 """
df = pd.DataFrame(
    results,
    columns=["Vk", "phiK", "p1", "p2"]
)
dirname = cfs.getFilename(
    ('edcK',theta_deviation,nShells,Vg,phiG,w1p,w1d,spreadE),
    dirname=utils.get_home_dn(machine)+"Data/",
    floatPrecision=3,
) + '_' + listFn + '/'
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




















