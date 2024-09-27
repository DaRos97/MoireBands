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
import functions as fs
from PIL import Image
from pathlib import Path

"""
We need to compute the interlayer coupling to modify the shape of the band mostly close to Gamma.
"""

#BZ cut parameters
cut = 'KGK'
n_pts = 301
K_list = fs.get_K(cut,n_pts)
K_scalar = np.zeros(K_list.shape[0])
for i in range(K_list.shape[0]):
    K_scalar[i] = np.linalg.norm(K_list[i])
#Extract S11 image
machine = cfs.get_machine(cwd)
S11_fn = fs.get_S11_fn(machine)
K = -K_list[0,0]
EM = -0.5
Em = -2.5
bounds = (K,EM,Em)
pic = fs.extract_png(S11_fn,[-K,K,EM,Em])

DFT = False
txt = 'DFT' if DFT else 'fit'
#TB paramaters
pars_mono = {}
hopping = {}
epsilon = {}
HSO = {}
par_offset = {}
for TMD in cfs.TMDs:
    pars_mono[TMD] = np.load(fs.get_pars_fn(TMD,machine,DFT))
    hopping[TMD] = cfs.find_t(pars_mono[TMD])
    epsilon[TMD] = cfs.find_e(pars_mono[TMD])
    HSO[TMD] = cfs.find_HSO(pars_mono[TMD][-2:])
    par_offset[TMD] = pars_mono[TMD][-3]

if 1:   #Final plot
    best_pars = {
            'DFT':{
                'no': (0,0,0,-0.5),
                'U1': (1,0.7,0.7,-0.5),
                'C6': (0.1,0.29,0.65,-0.5),
                'C3': (0,0.33,0.75,-0.5),
                },
            'fit':{
                'no': (0,0,0,-0.48),
                'U1': (1,0.9,0.88,-0.48),
                'C6': (0.15,0.32,0.75,-0.48),
                'C3': (0,0.35,0.8,-0.48),
                }
            }
    #plot
    fig,ax = plt.subplots()
    fig.set_size_inches(14,7)
    #Background
    ax.imshow(pic)
    #Different interlayers
    colors = {'no':'r','U1':'b','C6':'g','C3':'m'}
    legend_elements = []
    ens = {}
    for int_type in best_pars[txt].keys():
        energies = fs.energy(K_list,hopping,epsilon,HSO,par_offset,best_pars[txt][int_type],int_type)
        for i in range(22,30):
            ax.plot((K_list[:,0]+K)/2/K*pic.shape[1],(EM-energies[:,i])/(EM-Em)*pic.shape[0],color=colors[int_type])
        legend_elements.append(Line2D([0],[0],ls='-',color=colors[int_type],label=int_type,linewidth=1))
        ens[int_type] = np.copy(energies)
    ax.legend(handles=legend_elements,loc='upper right',fontsize=20)
    
    ax.set_xticks([0,pic.shape[1]//2,pic.shape[1]],[r"$K'$",r'$\Gamma$',r'$K$'],size=20)
    ax.set_yticks([0,pic.shape[0]//2,pic.shape[0]],["{:.2f}".format(EM),"{:.2f}".format((EM+Em)/2),"{:.2f}".format(Em)])
    ax.set_ylabel("$E\;(eV)$",size=20)
    ax.set_ylim(pic.shape[0],0)
    ax.set_title(txt,size=20)
    plt.show()
    if input("Save?[y/N]")=='y':
        fig.savefig('results/figures/'+txt+'.png')
        for int_type in best_pars[txt].keys():
            np.save('results/'+txt+'_'+int_type+'_pars_interlayer.npy',np.array(best_pars[txt][int_type]))
        for int_type in ['C6','C3']:
            fname = 'results/Data_GM/EvsK_bilayer_'+int_type+'.txt'
            savefile = np.zeros((K_list.shape[0],6))
            savefile[:,0] = K_list[:,0]
            savefile[:,1] = K_list[:,1]
            for nn in range(24,28):
                savefile[:,2+nn-24] = ens[int_type][:,nn]
            np.savetxt(fname,savefile,fmt='%.6e',delimiter='\t',
                        header='The six columns are: kx,ky,energy band lowest energy to highest.'
                    )
            fname2 = 'results/Data_GM/interlayer_pars_bilayer_'+int_type+'.txt'
            savefile2 = np.array(best_pars['fit'][int_type])
            np.savetxt(fname2,savefile2,fmt='%.3e',delimiter='\t',
                    header='The 4 rows are paramters: a, b, c, offset.'
                    )











