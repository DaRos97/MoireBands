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

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import functions2 as fs2
from PIL import Image
from pathlib import Path

"""
We need to compute the interlayer coupling to modify the shape of the band mostly close to Gamma.
"""

machine = cfs.get_machine(cwd)
sample = 'S11' if len(sys.argv) == 1 else sys.argv[1]
DFT = True
#BZ cut parameters
cut = 'Kp-G-K'
k_pts = 300
K_list = cfs.get_K(cut,k_pts)
K0 = np.linalg.norm(K_list[0])

sample_fn = fs2.get_sample_fn(sample,machine)
energy_bounds = {'S11': (-0.5,-2.5), 'S3': (-0.2,-1.8)}
EM, Em = energy_bounds[sample]
pic = fs2.extract_png(sample_fn,[-K0,K0,EM,Em],sample)
#
txt = 'DFT' if DFT else 'fit'
#TB paramaters
pars_mono = {}
hopping = {}
epsilon = {}
HSO = {}
par_offset = {}
for TMD in cfs.TMDs:
    DFT_1 = DFT if TMD=='WSe2' else True
    pars_mono[TMD] = np.load(fs2.get_pars_fn(TMD,machine,DFT_1))
    if not DFT_1:
        pars_mono[TMD] = np.append(pars_mono[TMD],np.load(fs2.get_SOC_fn(TMD,machine)))
    hopping[TMD] = cfs.find_t(pars_mono[TMD])
    epsilon[TMD] = cfs.find_e(pars_mono[TMD])
    HSO[TMD] = cfs.find_HSO(pars_mono[TMD][-2:])
    par_offset[TMD] = pars_mono[TMD][-3]

dic_offset = {'S11': (-0.5,-0.5), 'S3': (-0.03,-0.03)}
offset = dic_offset[sample]
best_interlayer_pars = {
        'DFT':{
            'no': (0,0,0,offset[0]),
            'U1': (1,0.7,0.7,offset[0]),
            #'C6': (0.1,0.29,0.65,offset[0]),
            'C6': (0,0.165,0.75,offset[0]),
            'C3': (0,0.33,0.75,offset[0]),
            },
        'fit':{
            'no': (0,0,0,offset[1]),
            'U1': (1,0.7,0.7,offset[1]),
            #'C6': (0.15,0.32,0.75,offset[1]),
            'C6': (0,0.165,0.75,offset[1]),
            'C3': (0,0.33,0.75,offset[1]),
            }
        }
#plot
fig,ax = plt.subplots()
fig.set_size_inches(14,7)
#Background
ax.imshow(pic)
#Different interlayers
colors = {'no':'r','U1':'m','C6':'g','C3':'b'}
legend_elements = []
label = {'no':'bare','C6':'interlayer C6','C3':'interlayer C3'}
ens = {}
for int_type in ['no','C6','C3']:#best_interlayer_pars[txt].keys():
    energies = fs2.energy(K_list,hopping,epsilon,HSO,par_offset,best_interlayer_pars[txt][int_type],int_type)
    for i in range(22,30):  #22-26
        ax.plot(np.arange(k_pts)/k_pts*pic.shape[1],(EM-energies[:,i])/(EM-Em)*pic.shape[0],color=colors[int_type])
    legend_elements.append(Line2D([0],[0],ls='-',color=colors[int_type],label=label[int_type],linewidth=1))
    ens[int_type] = np.copy(energies)
ax.legend(handles=legend_elements,loc='upper center',fontsize=25)

ax.set_xticks([0,pic.shape[1]//2,pic.shape[1]],[r"$K'$",r'$\Gamma$',r'$K$'],size=25)
ax.set_yticks([0,pic.shape[0]//2,pic.shape[0]],["{:.2f}".format(EM),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(Em)])
ax.set_ylabel("Energy (eV)",size=25)
ax.set_ylim(pic.shape[0],0)
ax.set_xlim(0,pic.shape[1])
fig.tight_layout()
#ax.set_title(txt,size=20)
plt.show()


exit()
if input("Save?[y/N]")=='y':
    title = sample+'_'+txt
    fig.savefig('results/figures/'+title+'.png')
    for int_type in best_interlayer_pars[txt].keys():
        res_fn = fs2.get_res_fn(title,int_type,machine)
        np.save(res_fn,np.array(best_interlayer_pars[txt][int_type]))











