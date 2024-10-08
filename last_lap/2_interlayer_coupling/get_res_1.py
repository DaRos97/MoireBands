import sys,os
import numpy as np
cwd = os.getcwd()
master_folder = cwd[:43]
sys.path.insert(1, master_folder)
import CORE_functions as cfs
tb_folder = master_folder+'/1_tight_binding'
sys.path.insert(2, tb_folder)
import functions1 as fs1
import functions2 as fs2

"""
Get tb pars from step 1
"""
machine = cfs.get_machine(cwd)

#DFT and SOC
for TMD in cfs.TMDs:
    pars = cfs.initial_pt[TMD]  #DFT values
    np.save(fs2.get_home_dn(machine)+'inputs/pars_'+TMD+'_DFT.npy',pars)
    np.save(fs2.get_SOC_fn(TMD,machine),np.load(fs1.get_SOC_fn(TMD,machine)))
#fit
inds_best = [12,29]
ind_reduced = 7
for i in inds_best:
    spec_args = fs1.get_spec_args(i) + (ind_reduced,)
    res_fn = fs1.get_res_fn(spec_args,machine)
    print("importing: ",res_fn)
    full_pars = np.load(res_fn)
    np.save(fs2.get_pars_fn(spec_args[0],machine,False),full_pars)
