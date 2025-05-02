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
plt.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D
import functions2 as fs2
from pathlib import Path
import itertools

"""
We need to compute the interlayer coupling to modify the shape of the band mostly close to Gamma.
"""

machine = cfs.get_machine(cwd)
sample = 'S11' if len(sys.argv) == 1 else sys.argv[1]
monolayer_type = 'fit'
#BZ cut parameters
cut = 'Kp-G-K'
k_pts = 100
K_list = cfs.get_K(cut,k_pts)
K0 = np.linalg.norm(K_list[0])

sample_fn = fs2.get_sample_fn(sample,machine)
EM, Em = cfs.dic_energy_bounds[sample]
pic = fs2.extract_png(sample_fn,[-K0,K0,EM,Em],sample)

#import TB paramaters
pars_mono = {}
hopping = {}
epsilon = {}
HSO = {}
par_offset = {}
for TMD in cfs.TMDs:
    pars_mono[TMD] = np.load(fs2.get_pars_fn(TMD,machine,monolayer_type))
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
            'C3': (0,0.344,0.6,offset[1]),
            }
        }
}
#plot
fig,ax = plt.subplots(figsize=(20,15))
#Background
ax.imshow(pic)
#Different interlayers
colors = {'no':'r','U1':'m','C6':'g','C3':'b'}
legend_elements = []
label = {'no':'bare','C6':'interlayer C6','C3':'interlayer C3'}
ens = {}
if False:    #fit data by hand
    tttt = 'C6'
    par_a = np.linspace(0.15,0.2,1)
    par_b = np.linspace(0.1,0.2,3)
    par_c = np.linspace(0.2,0.8,4)
    ind = int(sys.argv[2])
    a_,b_,c_ = list(itertools.product(*[par_a,par_b,par_c]))[ind]
    best_interlayer_pars[sample]['fit'][tttt] = (a_,b_,c_,offset[1])
    text = tttt+" - "+" a:"+"{:.4f}".format(a_)+", b:"+"{:.4f}".format(b_)+", c:"+"{:.4f}".format(c_)
    print(text)
    fig_fn = "results/figures/temp/"+text+'.png'
else:
    text = sample
    fig_fn = "results/figures/"+text+'_'+monolayer_type+'.png'
for int_type in ['no','C6','C3']:
    energies = fs2.energy(K_list,hopping,epsilon,HSO,par_offset,best_interlayer_pars[sample][monolayer_type][int_type],int_type)
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
ax.set_title(text,size=30)
fig.tight_layout()
#ax.set_title(txt,size=20)
plt.savefig(fig_fn)
#plt.show()

if input("Save?[y/N]")=='y':
    title = sample+'_'+monolayer_type
#    fig.savefig('results/figures/'+title+'.png')
    for int_type in ['no','C6','C3']:
        res_fn = fs2.get_res_fn(title,int_type,machine)
        np.save(res_fn,np.array(best_interlayer_pars[sample][monolayer_type][int_type]))











