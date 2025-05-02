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

#fit
P = 0.1
rp = 0.5
rl = 0
ind_reduced = 14
Pbc = 10
Pdk = 20
for TMD in cfs.TMDs:
    #DFT
    pars = cfs.initial_pt[TMD]  #DFT values
    np.save(fs2.get_pars_fn(TMD,machine,'DFT'),pars)
    #Fit and SOC
    SOC_pars = np.load(fs1.get_SOC_fn(TMD,machine))
    spec_args = (TMD,P,rp,rl,ind_reduced,Pbc,Pdk)
    res_fn = fs1.get_res_fn(spec_args,machine)
    print("importing: ",res_fn)
    full_pars = np.array(np.append(list(np.load(res_fn)),SOC_pars[-2:]))
    np.save(fs2.get_pars_fn(TMD,machine,'fit'),full_pars)
