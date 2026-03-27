"""
Here we take an intensity cut at Gamma or K to look at the distance between the main band and the side band crossing below it.
Gamma:
    For the 4-dimensional space of V, phi, w1p and w1d we define a distance from the ARPES result in two ways:
        - Considering distances: d1,d2 distance of side band crossing(sbc) and of WS2 band from TVB, respectively
        - Considering positions: p1,p2,p3 positions of TVB, sbc and WS2, respectively.
    From the ARPES exp we have, for S11 [eV]:
        - d1=0.0932, d2=0.6601
        - p1=-1.1599, p2=-1.2531, p3=-1.8200
K:
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
    master_folder = cwd[:20] + '/git/MoireBands/Code/'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code/'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import utils
import pandas as pd
from pathlib import Path

machine = cfs.get_machine(os.getcwd())
n_chunks = 128

""" Parameters and options """
if len(sys.argv)!=3:
    print("Usage: python3 edc.py arg1 arg2\nWith: arg1 in {'G','K'}, arg2 index")
    exit()
sample = 'S11'
BZpoint = sys.argv[1]
if BZpoint not in ['G','K']:
    raise ValueError("Not recognized BZ point: ",BZpoint)
ind = int(sys.argv[2])
if machine=='maf':
    ind -= 1
if ind<0 or ind>=n_chunks:
    raise ValueError("Index out of range: ",ind)
disp = machine=='loc'

""" Fixed parametetrs """
theta_deviation = 0      #Change here for \pm 0.3 degrees
nShells = 2
if BZpoint=='G':
    Vk,phiK = (0.0077,106/180*np.pi)
    kList = np.array([np.zeros(2),])
    columns = ["Vg", "phiG", "w1p", "w1d", "p1", "p2", "p3"]
    argsFn = (Vk,phiK)
elif BZpoint=='K':
    Vg,phiG = (0.017,174/180*np.pi)
    w1p = -5.880
    w1d = 0.480
    kList = np.array([[4*np.pi/3/cfs.dic_params_a_mono['WSe2'],0],])
    columns=["Vk", "phiK", "p1", "p2"]
    argsFn = (Vg,phiG,w1p,w1d)
spreadE = 0.03      # in eV
#
nCells = cfs.get_nCells(nShells)
monolayer_fns = {
    'WSe2':master_folder+'Inputs/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy',
    'WS2':master_folder+'Inputs/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy'
}
theta = cfs.dic_params_twist[sample] + theta_deviation    #twist angle, in degrees, from LEED eperiment
stacking = 'P'
w2p = w2d = 0
if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_fns)
    print("Sample ",sample," with twist %.2f°"%theta," (variation of %.1f° from LEED)"%theta_deviation)
    if BZpoint=='G':
        print("Moiré potential at K (%.5f eV, %.1f°)"%(Vk,phiK/np.pi*180))
    elif BZpoint=='K':
        print("Moiré potential at G (%.5f eV, %.1f°)"%(Vg,phiG/np.pi*180))
        print("Interlayer coupling: w1p=%.4f, w1d=%.4f"%(w1p,w1d))
    print("Number of mini-BZs circles: ",nShells)
    print("Energy spreading: %.3f eV"%spreadE)

""" Computation """
parameters_chunk, listFn = utils.get_parameters(ind,BZpoint,n_chunks=n_chunks)
results = []
for pars in parameters_chunk:
    if BZpoint=='G':
        Vg, phiG, w1p, w1d = pars
        if disp:
            print("Vg: %.3f\tphiG: %.1f\tw1p: %.3f\t w1d: %.3f"%(Vg,phiG/np.pi*180,w1p,w1d))
    elif BZpoint=='K':
        Vk, phiK = pars
        if disp:
            Vk = 0.0085
            phiK = 106/180*np.pi
            print("Vk: %.3f\tphiK: %.1f"%(Vk,phiK/np.pi*180))
    parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
    args_diag = (nShells, nCells, kList, monolayer_fns, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
    positions,success = utils.EDC(
        args_diag,
        sample,
        BZpoint=BZpoint,
        spreadE=spreadE,
        machine=machine,
        plotBands=False,
        plotFit=disp
    )
    if success:
        if BZpoint=='G':
            results.append((Vg,phiG,w1p,w1d,*positions))
        elif BZpoint=='K':
            results.append((Vk,phiK,*positions))
    else:
        if BZpoint=='G':
            results.append((Vg,phiG,w1p,w1d,np.nan,np.nan,np.nan))
        elif BZpoint=='K':
            results.append((Vk,phiK,np.nan,np.nan))

""" Save to file: hdf5 """
df = pd.DataFrame(
    results,
    columns=columns
)
dirname = cfs.getFilename(
    ('edc'+BZpoint,theta_deviation,nShells,spreadE,*argsFn),
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

print("Finished %s chunk %d/%d"%(BZpoint,ind,n_chunks))




















