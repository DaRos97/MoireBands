import numpy as np
import sys,os
import functions as fs

machine = fs.get_machine(os.getcwd())

"""
Copy eye pars pars in 3_moire/inputs/
"""
if 1:
    pars_interlayer = np.load(fs.get_home_dn(machine)+'results/pars_interlayer.npy')
    np.save('/home/dario/Desktop/git/MoireBands/last_lap/3_moire/inputs/pars_interlayer.npy',pars_interlayer)

