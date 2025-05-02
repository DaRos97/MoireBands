import functions as fs
import sys
import numpy as np
import getopt
from time import time as tt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import tqdm 

dirname = "moire_Data/"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "E:",["LL=","UL=","EnGrid=","mono","offset_energy="])
    E = 0
    upper_layer = 'WSe2'
    lower_layer = 'WS2'
    offset_energy = -0.41
    mono = False
    gridy = 0
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-E']:
        E = float(arg)
    if opt == '--LL':
        lower_layer = arg
    if opt == '--UL':
        upper_layer = arg
    if opt == '--EnGrid':
        gridy = int(arg)
    if opt == '--mono':
        mono = True
    if opt == '--offset_energy':
        offset_energy = float(arg)

if gridy == 0:
    gridy = 400

N = 6
bands = int(1+3*N*(N+1))*2        #dimension of H divided by 3 -> take only valence bands     #Full Diag -> *3
energies = np.zeros((400,2,400,bands))
weights = np.zeros((400,2,400,bands))
mono_UL = np.zeros((400,400,6))
mono_LL = np.zeros((400,400,6))
for J in range(400):
    data_name = dirname+"energies_"+lower_layer+"-"+upper_layer+"_"+str(J)+".npy"
    weights_name = dirname+"arpes_"+lower_layer+"-"+upper_layer+"_"+str(J)+".npy"    
    mono_LL_name = dirname+"mono_"+lower_layer+'_'+str(J)+".npy"
    mono_UL_name = dirname+"mono_"+upper_layer+'_'+str(J)+".npy"
    energies[J] = np.load(data_name)
    weights[J] = np.load(weights_name)
    mono_UL[J] = np.load(mono_UL_name)
    mono_LL[J] = np.load(mono_LL_name)

energy_cut = 200
res = energies[:,:,energy_cut,:]
weight = weights[:,:,energy_cut,:]

#parameters of Lorentzian
lp = 400;     gridx = lp;    #grid in momentum fixed by points evaluated previously 
K_ = 0.0004      #spread in momentum
K2 = K_**2
E_ = 0.005       #spread in energy in eV
E2 = E_**2
min_e = np.amin(np.ravel(res[:,:bands,:]))
max_e = np.amax(np.ravel(res[:,:bands,:]))
larger_E = 0.2      #in eV. Enlargment of E axis wrt min and max band energies
MIN_E = min_e - larger_E
MAX_E = max_e + larger_E
delta = MAX_E - MIN_E
step = delta/gridy
#K-axis
Ki, Km, Kf = (np.array([-1,0]),np.array([0,0]),np.array([1,0]))
K_list = np.linspace(-np.linalg.norm(Ki-Km),np.linalg.norm(Kf-Km),lp)
E_list = np.linspace(MIN_E,MAX_E,gridy)
#Compute values of lorentzian spread of weights
lor_name = "Data/FC_"+lower_layer+"-"+upper_layer+"_"+str(energy_cut)
par_name = '_Full_('+str(gridy)+'_'+str(larger_E).replace('.',',')+'_'+str(K_).replace('.',',')+'_'+str(E_).replace('.',',')+')'+".npy"
lor_name += par_name
try:
    lor = np.load(lor_name)
except:
    print("\nComputing Lorentzian spread ...")
    lor = np.zeros((lp,gridy))
    for l in range(2):
        for i in tqdm.tqdm(range(lp)):
            for j in range(bands):
                pars = (K2,E2,weight[i,l,j],K_list[i],res[i,l,j])
                lor += fs.lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
    np.save(lor_name,lor)
## Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
if mono:
    for b in range(2):      #plot valence bands (2 for spin-rbit) of monolayer
        plt.plot(K_list,res_mono_LL[:,b],'r-',lw = 0.5)
    for b in range(2):      #plot valence bands (2 for spin-rbit) of monolayer
        plt.plot(K_list,res_mono_UL[:,b]-offset_energy,'r-',lw = 0.5)
#for i,c in enumerate([*Path]):      #plot symmetry points as vertical lines
#    a = 1 if i == 2 else 0
#    plt.vlines(K_list[i*lp//2-a],MIN_E,MAX_E,'k',lw=0.3,label=c)
#    plt.text(K_list[i*lp//2-a],MIN_E-delta/12,dic_sym[c])
#
X,Y = np.meshgrid(K_list,E_list)
plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=lor[np.nonzero(lor)].min(), vmax=lor.max()))
plt.ylabel('eV')
plt.show()

























