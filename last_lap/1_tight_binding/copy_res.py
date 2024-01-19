import numpy as np
import parameters as ps
import functions as fs
import sys

machine = fs.get_machine(os.getcwd())
TMD = sys.argv[1]

"""
Copy DFT pars in 2_* inputs/
"""
if 1:
    for TMD in ['WSe2','WS2']:
        pars = ps.initial_pt[TMD]  #DFT values
        np.save('/home/dario/Desktop/git/MoireBands/last_lap/2_interlayer_coupling/inputs/pars_'+TMD+'.npy',pars)

"""
Copy result of minimization in 2_* inputs
"""
if 0:
    os.system('cp '+fs.get_home_dn(machine)+'results/*.npy /home/dario/Desktop/git/MoireBands/last_lap/2_interlayer_coupling/inputs/')

