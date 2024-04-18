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
vals = [('WSe2',0.4,True),
        ('WS2',0.2,True)
        ]
for val in vals:
    TMD, range_par, fixed_SO = val
    pars = [0]
    for file in os.listdir('/home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/'):
        terms = file.split('_')
        if terms[1] == TMD and terms[2]=="{:.2f}".format(range_par).replace('.',',') and str(fixed_SO)==terms[3]:
            pars = np.load('/home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/'+file)
    if len(pars)==1:
        print("Parameters not found for TMD: ",TMD,", range_par: ",range_par," and fixed SO: ",str(fixed_SO))
        continue

    if fixed_SO:
        SO_values = ps.initial_pt[TMD][-2:]
        full_pars = list(pars)
        for i in range(2):
            full_pars.append(SO_values[i])
    else:
        full_pars = pars

    np.save(fs.get_home_dn(machine)+'inputs/pars_'+TMD+'_fit.npy',full_pars)
