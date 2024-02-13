import numpy as np
import matplotlib.pyplot as plt
import functions as fs
from PIL import Image
import os

"""
We need to compute the interlayer coupling to modify the shape of the band mostly close to Gamma.
"""
#BZ cut parameters
cut = 'KGK'
n_pts = 200
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
#Interlayer type
interlayer_type = 'C3'

for DFT in [True]:#,False]:
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
        par_offset[TMD] = pars_mono[TMD][-1]
    if 1:   #minimization of interlayer based on first 3 bands
        for a in [0.]:#np.linspace(0,0.1,3):
            for b in np.linspace(0.3,0.4,5):
                for c in np.linspace(0.6,0.8,5):
                    for offset in [-0.5]:
                        pars_interlayer = (a,b,c,offset)
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
                        fig.savefig('results/temp/'+txt+'_'+interlayer_type+'_'+"{:.2f}".format(a)+'_'+"{:.2f}".format(b)+'_'+"{:.2f}".format(c)+'_'+"{:.2f}".format(offset)+'.png')
#                        fig.savefig('temp/fig_'+"{:.2f}".format(a)+'_'+"{:.2f}".format(b)+'_'+"{:.2f}".format(c)+'_'+"{:.2f}".format(offset)+'.png')
                        plt.close(fig)
        continue
    #No Interlayer
    a = 0
    b = 0
    c = 0
    offset = -0.5
    pars_interlayer = (a,b,c,offset)
    energies = fs.energy(K_list,hopping,epsilon,HSO,par_offset,pars_interlayer,'no')
    #plot
    title = "S11 data with monolayer "+txt+" bands and no interlayer"
    fig = fs.plot_bands_on_exp(energies,pic,K_list,bounds,title)
    fig.savefig('results/figures/'+txt+'_no-int_'+"{:.2f}".format(a)+'_'+"{:.2f}".format(b)+'_'+"{:.2f}".format(c)+'_'+"{:.2f}".format(offset)+'.png')
    #Interlayer
    if DFT:
        if interlayer_type == 'U1':
            pars_interlayer = (1,0.7,0.7,-0.5)
        if interlayer_type == 'C6':
            pars_interlayer = (0.1,0.29,0.65,-0.5)
            #pars_interlayer = (0.,0.25,0.73,-0.5)
        if interlayer_type == 'C3':
            pars_interlayer = (0,0.33,0.75,-0.5)
    else:
        if interlayer_type == 'U1':
            pars_interlayer = (0.7,0.7,0.9,-0.48)
    a,b,c,offset = pars_interlayer
    energies = fs.energy(K_list,hopping,epsilon,HSO,par_offset,pars_interlayer,interlayer_type)
    title = "S11 data with monolayer "+txt+" bands and interlayer "+interlayer_type
    fig = fs.plot_bands_on_exp(energies,pic,K_list,bounds,title)
    fig.savefig('results/figures/'+txt+'_'+interlayer_type+'_'+"{:.2f}".format(a)+'_'+"{:.2f}".format(b)+'_'+"{:.2f}".format(c)+'_'+"{:.2f}".format(offset)+'.png')
    pars_interlayer = np.array([a,b,c,offset])
    np.save(fs.get_home_dn(machine)+'results/'+txt+'_'+interlayer_type+'_pars_interlayer.npy',pars_interlayer)


