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
#TB paramaters
machine = fs.get_machine(os.getcwd())
pars_mono = {}
hopping = {}
epsilon = {}
HSO = {}
offset = {}
for TMD in fs.materials:
    pars_mono[TMD] = np.load(fs.get_pars_fn(TMD,machine))
    hopping[TMD] = fs.find_t(pars_mono[TMD])
    epsilon[TMD] = fs.find_e(pars_mono[TMD])
    HSO[TMD] = fs.find_HSO(pars_mono[TMD])
    offset[TMD] = pars_mono[TMD][-1]
#Extract S11 image
S11_fn = fs.get_S11_fn(machine)
K = 4/3*np.pi/fs.dic_params_a_mono['WSe2']
EM = -0.5
Em = -2.5
bounds = (K,EM,Em)
pic = fs.extract_png(S11_fn,[-K,K,EM,Em])

if 0:   #Compare S11 bilayer KGK with monolayer bands
    pars_interlayer = (0,0,0)
    global_offset = 0.3
    energies = fs.energy(K_list,hopping,epsilon,HSO,offset,pars_interlayer,global_offset)
    #plot
    title = "S11 data with monolayer bands (no interlayer), offset="+"{:.2f}".format(global_offset)+" eV"
    fig = fs.plot_bands_on_exp(energies,pic,K_list,bounds,False,title)
    plt.show()
    exit()

if 1:   #minimization of interlayer based on first 3 bands
    for a in [1]:
        for b in [0.7]:
            for c in [0.75]:
                for global_offset in [0.25]:
                    pars_interlayer = (a,b,c)
                    energies = fs.energy(K_list,hopping,epsilon,HSO,offset,pars_interlayer,global_offset)
                    fig = fs.plot_bands_on_exp(energies,pic,K_list,bounds,True)
                    fig.savefig('temp/fig_'+"{:.2f}".format(a)+'_'+"{:.2f}".format(b)+'_'+"{:.2f}".format(c)+'_'+"{:.2f}".format(global_offset)+'.png')
                    plt.close(fig)



