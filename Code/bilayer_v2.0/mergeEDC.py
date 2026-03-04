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

""" Dirname and parameters """
machine = cfs.get_machine(os.getcwd())
n_chunks = 128
theta_deviation = 0      #Change here for \pm 0.3 degrees
nShells = 2
Vk,phiK = (0.006,106/180*np.pi)
spreadE = 0.03      # in eV
chunk, listFn = utils.get_parameters(0,n_chunks=n_chunks)
dirname = cfs.getFilename(
    ('edcGamma',theta_deviation,nShells,Vk,phiK,spreadE),
    dirname=utils.get_home_dn(machine)+"Data/",
    floatPrecision=3
) + listFn + '/'

files = sorted(glob.glob(dirname+"*_%d.h5"%n_chunks))

output_file = cfs.getFilename(
    ('full_edcGamma',theta_deviation,nShells,Vk,phiK,spreadE),
    dirname=utils.get_home_dn(machine)+"Data/",
    floatPrecision=3
) + listFn +  ".h5"

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

print("Merged %d chunk files."%len(files))
