"""
Here we include interlayer coupling in the bilayer to modify the shape of the bands at Gamma to coincide with the experiment.
The fitting has to be done separately for the 2 samples since they might have different stackings: S3 and S11.
`monolayer_type` specifies weather to use the DFT or fit values for the monolayer bands.
"""
import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_interlayer as fsi
from pathlib import Path
import itertools
machine = cfs.get_machine(cwd)

automated_comparison = 0#True

if len(sys.argv) not in [2,3]:
    print("Usage: py interlayer.py S3(S11) ind(optional)")
sample = sys.argv[1]
monolayer_type = 'fit'
if monolayer_type=='fit' and not automated_comparison:
    print("Make sure you imported the most recent fit parameters from step 1 and save them as `inputs/pars_*TMD*`.npy")
#BZ cut parameters
cut = 'Kp-G-K'
k_pts = 100
K_list = cfs.get_K(cut,k_pts)
K0 = np.linalg.norm(K_list[0])

sample_fn = fsi.get_sample_fn(sample,machine)
EM, Em = cfs.dic_energy_bounds[sample]
pic = fsi.extract_png(sample_fn,[-K0,K0,EM,Em],sample)

#import TB paramaters
pars_mono = {}
hopping = {}
epsilon = {}
HSO = {}
par_offset = {}
for TMD in cfs.TMDs:
    pars_mono[TMD] = np.load(fsi.get_fit_pars_fn(TMD,machine)) if monolayer_type=='fit' else np.array(cfs.initial_pt[TMD])
    hopping[TMD] = cfs.find_t(pars_mono[TMD])
    epsilon[TMD] = cfs.find_e(pars_mono[TMD])
    HSO[TMD] = cfs.find_HSO(pars_mono[TMD][-2:])
    par_offset[TMD] = pars_mono[TMD][-3]
#
dic_offset = {'S11': (-0.5,-0.5), 'S3': (-0.0,-0.0)}
offset = dic_offset[sample]
best_interlayer_pars = {
    'S11':{
        'DFT':{
            'no': (0,0,0,offset[0]),
            'U1': (1,0.7,0.7,offset[0]),
            'C6': (0,0.165,0.75,offset[0]),
            'C3': (0,0.33,0.75,offset[0]),
            },
        'fit':{
            'no': (0,0,0,offset[1]),
#            'U1': (1,0.7,0.7,offset[1]),
            'C6': (0.15,0.2,0.8,offset[1]),
            'C3': (0,0.34,0.8,offset[1]),
            }
        },
    'S3':{
        'DFT':{
            'no': (0,0,0,offset[0]),
            'U1': (1,0.7,0.7,offset[0]),
            #'C6': (0.1,0.29,0.65,offset[0]),
            'C6': (0,0.165,0.75,offset[0]),
            'C3': (0,0.33,0.75,offset[0]),
            },
        'fit':{
            'no': (0,0,0,offset[1]),
#            'U1': (1,0.7,0.7,offset[1]),
            'C6': (0.15,0.2,0.61,offset[1]),
            'C3': (0,0.32,0.66,offset[1]),
            }
        }
}

if automated_comparison:    #fit data by hand
    tttt = 'C3'
    par_a = np.linspace(0.,0.,1)
    par_b = np.linspace(0.28,0.32,5)
    par_c = np.linspace(0.6,0.72,5)
    ind = int(sys.argv[2])
    a_,b_,c_ = list(itertools.product(*[par_a,par_b,par_c]))[ind]
    best_interlayer_pars[sample][monolayer_type][tttt] = (a_,b_,c_,offset[1])
    text = tttt+" - "+" a:"+"{:.4f}".format(a_)+", b:"+"{:.4f}".format(b_)+", c:"+"{:.4f}".format(c_)
    print(text)
    fig_fn = "Figures/temp/"+text+'.png'
else:
    text = sample
    fig_fn = ''

args = (pic,K_list,hopping,epsilon,HSO,par_offset,best_interlayer_pars,sample,monolayer_type,k_pts,EM,Em)
fsi.plot_bands(*args,figname=fig_fn,show=(not automated_comparison),title=text)

if not automated_comparison:
    if input("Save?[y/N]")=='y':
        title = sample+'_'+monolayer_type
        fsi.plot_bands(*args,figname='Figures/'+title+'.png',show=False,title=title)
        for int_type in ['no','C6','C3']:
            res_fn = fsi.get_res_fn(title,int_type,machine)
            np.save(res_fn,np.array(best_interlayer_pars[sample][monolayer_type][int_type]))











