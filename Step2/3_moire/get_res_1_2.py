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

#fit
P = 0.1
rp = 0.5
rl = 0
ind_reduced = 14
Pbc = 10
Pdk = 20
for TMD in cfs.TMDs:
    #Fit and SOC
    SOC_pars = np.load(fs1.get_SOC_fn(TMD,machine))
    spec_args = (TMD,P,rp,rl,ind_reduced,Pbc,Pdk)
    res_fn = fs1.get_res_fn(spec_args,machine)
    print("importing tb parameters: ",res_fn)
    full_pars = np.array(np.append(list(np.load(res_fn)),SOC_pars[-2:]))
    np.save(fs3.get_pars_mono_fn(TMD,machine,'fit'),full_pars)

"""
Get interlayer pars from step 2
"""
for sample in ['S11','S3']:
    for monolayer_type in ['fit',]:
        title = sample+'_'+monolayer_type
        for int_type in ['C6','C3']:
            int_fn = fs2.get_res_fn(title,int_type,machine)
            #DFT values
            np.save(fs3.get_pars_interlayer_fn(sample,int_type,monolayer_type,machine),np.load(int_fn))







