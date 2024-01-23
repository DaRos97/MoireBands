import numpy as np
import sys,os
import functions as fs

sys.path.insert(1, '/home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding')
import parameters as ps

machine = fs.get_machine(os.getcwd())

"""
Get tb pars from step 1
"""
#DFT values
for TMD in ['WSe2','WS2']:
    pars = ps.initial_pt[TMD]  #DFT values
    np.save(fs.get_home_dn(machine)+'inputs/pars_'+TMD+'_DFT.npy',pars)

#Minimization values
vals = [('WSe2',0.2,['KGK','KMKp']),('WS2',0.2,['KGK','KMKp'])]
for val in vals:
    TMD, range_par, cuts = val
    cuts_fn = ''
    for i in range(len(cuts)):
        cuts_fn += cuts[i]
        if i != len(cuts)-1:
            cuts_fn += '_'
    pars = [0]
    for file in os.listdir('/home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/'):
        if file[5:5+len(TMD)] == TMD and file[6+len(TMD):10+len(TMD)]=="{:.2f}".format(range_par).replace('.',',') and file[11+len(TMD):11+len(TMD)+len(cuts_fn)]==cuts_fn:
            pars = np.load('/home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/'+file)
    if len(pars)==1:
        print("Parameters not found for TMD: ",TMD,", range_par: ",range_par," and cuts: ",cuts_fn)
        continue
    np.save(fs.get_home_dn(machine)+'inputs/pars_'+TMD+'.npy',pars)


"""
Get interlayer pars from step 2
"""
#DFT values
pars_dft = np.load('/home/dario/Desktop/git/MoireBands/last_lap/2_interlayer_coupling/results/pars_interlayer_dft.npy')
np.save(fs.get_home_dn(machine)+'inputs/pars_interlayer_dft.npy',pars_dft)
#Minimization values
pars = np.load('/home/dario/Desktop/git/MoireBands/last_lap/2_interlayer_coupling/results/pars_interlayer.npy')
np.save(fs.get_home_dn(machine)+'inputs/pars_interlayer.npy',pars)
