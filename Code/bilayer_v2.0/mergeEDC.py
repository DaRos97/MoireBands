import sys,os
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import glob
import pandas as pd
import numpy as np
import utils
from pathlib import Path

""" Input """
if len(sys.argv)!=2:
    print("Usage: python3 mergeEDC.py arg1\nWith arg1 = {'G','K'}")
    exit()
BZpoint = sys.argv[1]

""" Dirname and parameters """
machine = cfs.get_machine(os.getcwd())
n_chunks = 128
theta_deviation = 0      #Change here for \pm 0.3 degrees
nShells = 2
if BZpoint=='G':
    args = (0.0077,106/180*np.pi)        #Vk, phiK
else:
    args = (0.0165,175/180*np.pi,-1.556,1.143)        #Vg, phiG, w1p, w1d
spreadE = 0.03      # in eV

""" Filenames """
chunk, listFn = utils.get_parameters(0,BZpoint,n_chunks=n_chunks)
dirname = cfs.getFilename(
    ('edc'+BZpoint,theta_deviation,nShells,spreadE,*args),
    dirname=utils.get_home_dn(machine)+"Data/",
    floatPrecision=3
) + '_' + listFn + '/'

files = sorted(glob.glob(dirname+"*_%d.h5"%n_chunks))

output_file = cfs.getFilename(
    ('full_edc'+BZpoint,theta_deviation,nShells,spreadE,*args),
    dirname=utils.get_home_dn(machine)+"Data/",
    floatPrecision=3
) + '_' + listFn +  ".h5"

""" Loop and store """
store = pd.HDFStore(
    output_file,
    mode="w",
    complevel=5,
    complib="blosc"
)

for f in files:
    df = pd.read_hdf(f, key="results")
    store.append(
        "results",
        df,
        format="table",
        data_columns=True
    )

store.close()
print("Merged %d chunk files of edc%s."%(len(files),BZpoint))

# Look for gaps
dirnameGap = cfs.getFilename(
    ('edcGap'+BZpoint,theta_deviation,nShells,spreadE,*args),
    dirname=utils.get_home_dn(machine)+"Data/",
    floatPrecision=3
) + '_' + listFn + '/'
if Path(dirnameGap).is_dir():
    files = sorted(glob.glob(dirnameGap+"*_%d.h5"%n_chunks))

    output_file = cfs.getFilename(
        ('full_edcGap'+BZpoint,theta_deviation,nShells,spreadE,*args),
        dirname=utils.get_home_dn(machine)+"Data/",
        floatPrecision=3
    ) + '_' + listFn +  ".h5"

    """ Loop and store """
    store = pd.HDFStore(
        output_file,
        mode="w",
        complevel=5,
        complib="blosc"
    )

    for f in files:
        df = pd.read_hdf(f, key="results")
        store.append(
            "results",
            df,
            format="table",
            data_columns=True
        )

    store.close()
    print("With gaps")




