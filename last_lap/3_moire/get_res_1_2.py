import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
tb_folder = master_folder+'/1_tight_binding'
sys.path.insert(2, tb_folder)
import functions1 as fs1
int_folder = master_folder+'/2_interlayer_coupling'
sys.path.insert(3, int_folder)
import functions2 as fs2
import functions3 as fs3

"""
Get tb pars from step 1
"""
machine = cfs.get_machine(cwd)

#DFT and SOC
for TMD in cfs.TMDs:
    pars = cfs.initial_pt[TMD]  #DFT values
    np.save(fs3.get_home_dn(machine)+'inputs/pars_'+TMD+'_DFT.npy',pars)
    np.save(fs3.get_SOC_fn(TMD,machine),np.load(fs1.get_SOC_fn(TMD,machine)))
#fit
inds_best = [12,29]
ind_reduced = 7
for i in inds_best:
    spec_args = fs1.get_spec_args(i) + (ind_reduced,)
    res_fn = fs1.get_res_fn(spec_args,machine)
    print("importing: ",res_fn)
    full_pars = np.load(res_fn)
    np.save(fs3.get_pars_mono_fn(spec_args[0],machine,False),full_pars)


"""
Get interlayer pars from step 2
"""
for sample in ['S11','S3']:
    for txt in ['DFT','fit']:
        DFT = True if txt=='DFT' else False
        title = sample+'_'+txt
        for int_type in ['U1','C6','C3']:
            int_fn = fs2.get_res_fn(title,int_type,machine)
            #DFT values
            np.save(fs3.get_pars_interlayer_fn(sample,int_type,DFT,machine),np.load(int_fn))







