import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import functions as fs
from PIL import Image
from pathlib import Path
import os,sys

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
machine = fs.get_machine(os.getcwd())
S11_fn = fs.get_S11_fn(machine)
K = -K_list[0,0]
EM = -0.5
Em = -2.5
bounds = (K,EM,Em)
pic = fs.extract_png(S11_fn,[-K,K,EM,Em])

if 0:   #minimization of interlayer based on first 3 bands
    #Interlayer type -> 'U1','C6','C3'
    interlayer_type = 'C3'
    DFT = False

    #No sistematic proedure, for each set of parameters and interlayer type need to find the best by eye
    txt = 'DFT' if DFT else 'fit'
    print(txt)
    #TB paramaters
    pars_mono = {}
    hopping = {}
    epsilon = {}
    HSO = {}
    par_offset = {}
    for TMD in fs.materials:
        pars_mono[TMD] = np.load(fs.get_pars_fn(TMD,machine,DFT))
        hopping[TMD] = fs.find_t(pars_mono[TMD])
        epsilon[TMD] = fs.find_e(pars_mono[TMD])
        HSO[TMD] = fs.find_HSO(pars_mono[TMD])
        par_offset[TMD] = pars_mono[TMD][-3]
    for a in np.linspace(0,0.2,1):
        for b in np.linspace(0.3,0.5,5):
            for c in np.linspace(0.6,0.8,5):
                for offset in [-0.48]:
                    pars_interlayer = (a,b,c,offset) if not interlayer_type=='no' else (0,0,0,offset)
#                    pars_interlayer = (1.1,0.9,1.02,offset)
                    print(pars_interlayer)
                    figname = 'results/temp/'+txt+'_'+interlayer_type+'_'+"{:.2f}".format(a)+'_'+"{:.2f}".format(b)+'_'+"{:.2f}".format(c)+'_'+"{:.2f}".format(offset)+'.png'
                    if Path(figname).is_file():
                        print("Already computed")
                        continue
                    energies = fs.energy(K_list,hopping,epsilon,HSO,par_offset,pars_interlayer,interlayer_type)
                    if 0:
                        for i in range(24,28):
                            plt.plot(K_list[:,0],energies[:,i],'r-')
                        plt.ylim(Em,EM)
                        plt.show()
                        exit()
                    title = "S11 data with monolayer "+txt+" bands and interlayer "+interlayer_type
                    fig = fs.plot_bands_on_exp(energies,pic,K_list,bounds,title)
                    if 0:
                        plt.show()
                        exit()
                    fig.savefig(figname)
#                        fig.savefig('temp/fig_'+"{:.2f}".format(a)+'_'+"{:.2f}".format(b)+'_'+"{:.2f}".format(c)+'_'+"{:.2f}".format(offset)+'.png')
                    plt.close(fig)
        continue
    exit()

if 1:   #Final plot
    DFT = False
    txt = 'DFT' if DFT else 'fit'
    #TB paramaters
    pars_mono = {}
    hopping = {}
    epsilon = {}
    HSO = {}
    par_offset = {}
    for TMD in fs.materials:
        pars_mono[TMD] = np.load(fs.get_pars_fn(TMD,machine,DFT))
        hopping[TMD] = fs.find_t(pars_mono[TMD])
        epsilon[TMD] = fs.find_e(pars_mono[TMD])
        HSO[TMD] = fs.find_HSO(pars_mono[TMD])
        par_offset[TMD] = pars_mono[TMD][-3]
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
    from matplotlib.lines import Line2D
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
        for i in range(24,28):
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











