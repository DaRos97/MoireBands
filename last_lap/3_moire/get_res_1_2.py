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
import functions as fs

"""
Get tb pars from step 1
"""
machine = cfs.get_machine(cwd)

def get_spec_args_txt(spec_args):
    return "{:.3f}".format(spec_args[0]).replace('.',',')+'_'+"{:.3f}".format(spec_args[1]).replace('.',',')+'_'+"{:.3f}".format(spec_args[2]).replace('.',',')+'_'+"{:.3f}".format(spec_args[3]).replace('.',',')

spec_args_dic = {'WSe2': (0.11,1.,0.3,0),'WS2': (0.11,1.,0.3,0)}
#DFT values
for TMD in cfs.TMDs:
    pars = cfs.initial_pt[TMD]  #DFT values
    np.save(fs.get_home_dn(machine)+'inputs/pars_'+TMD+'_DFT.npy',pars)
    #
    fn = '../1_tight_binding/results/'+'res_'+TMD+'_'+get_spec_args_txt(spec_args_dic[TMD])+'.npy'
    print(fn)
    full_pars = np.load(fn)
    np.save(fs.get_home_dn(machine)+'inputs/pars_'+TMD+'_fit.npy',full_pars)


"""
Get interlayer pars from step 2
"""
for sample in ['S11','S3']:
    for txt in ['DFT','fit']:
        for interlayer_type in ['U1','C6','C3']:
            int_dn = '/home/dario/Desktop/git/MoireBands/last_lap/2_interlayer_coupling/results/'
            int_fn = sample+'_'+txt+'_'+interlayer_type+'_pars_interlayer.npy'
            #DFT values
            np.save(fs.get_home_dn(machine)+'inputs/'+int_fn,np.load(int_dn+int_fn))







