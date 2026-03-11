""" Plot of Figure 2: Bilayer with Interlayer coupling AND moirè.
Path: G -> K -> Kp -> G -> K -> Kp
    - Center around Gamma
    - Center around M
"""

import numpy as np
import sys, os
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions as fs
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


machine = cfs.get_machine(os.getcwd())
saveE = True
saveW = True

sample = 'S11'

""" Interlayer parameters """
theta = 2.8     # twisting angle, in deg
Vg = 0.017              # eV
phiG = 175/180*np.pi        # rad
Vk = 0.006              # eV
phiK = -106/180*np.pi       # rad
w1p = -1.66         # eV
w1d = 0.324         # eV
stacking = 'P'
w2p = w2d = 0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}

nShells = 2
nCells = int(1+3*nShells*(nShells+1))
monolayer_type = 'fit'
moire_pars = (Vg,Vk,phiG,phiK)
""" BZ path """
kPts = 200
kList,norm = cfs.get_kList('G-K-Kp',kPts,returnNorm=True)
kPts = kList.shape[0]
#K0 = np.linalg.norm(kList[-1])   #val of |K|
""" Computing evals and evecs """
args_e_data = (sample,nShells,monolayer_type,Vk,phiK,theta,stacking,w1p,w1d,phiG,kPts)
th_e_data_fn = 'Data/fig3_e_'+fs.get_fn(*args_e_data)+'.npz'
if not Path(th_e_data_fn).is_file():
    print("Computing")
    args = (nShells, nCells, kList, monolayer_type, parsInterlayer, theta, moire_pars, '', False, True)
    evals, evecs = fs.diagonalize_matrix(*args)
    if saveE:
        np.savez(th_e_data_fn,evals=evals,evecs=evecs,norm=norm,kList=kList)
else:
    print("Loading energies")
    evals = np.load(th_e_data_fn)['evals']
    evecs = np.load(th_e_data_fn)['evecs']
    norm = np.load(th_e_data_fn)['norm']

""" Spread parameters """
spreadK = 0.005     # in 1/a
spreadE = 0.01      # in eV
typeSpread = 'Gauss'    # 'Gauss' or 'Lorentz', works for both k and E
deltaE = 0.01       # in eV, sets the energy grid
shadeFactor = 0.1     # shade factor of WS2 -> 0 (not visible at all) to 1 (same relevance as WSe2)
# we could also put another shading depending on the energy -> lower bands get less weight
powFactor = 1.       # exponent of weights -> 2 is the usual mod square of eigenvectors which should be the weight of ARPES spectra. For lower values we enhance the intensity of the side bands
minimumBand = 15    # lowest considered band for spreading      #bands are 0 to 43, with TVB at 27
pars_spread = ( spreadK, spreadE, typeSpread, deltaE, powFactor, shadeFactor, minimumBand)
""" Computing weights and spread image """
args_w_data = args_e_data + pars_spread
th_w_data_fn = 'Data/fig3_w_'+fs.get_fn(*args_w_data)+'.npz'
if not Path(th_w_data_fn).is_file():
    print("Computing weight and spread")
    weights = np.zeros((kPts,nCells*44))
    for i in range(kPts):
        ab = np.absolute(evecs[i])**powFactor
        weights[i,:] = np.sum(ab[:22,:],axis=0) + shadeFactor*np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    E_max, E_min, pKi, pKf, pEmax, pEmin = cfs.dic_pars_samples[sample]
    E_min = -2.5        #bit smaller than exp image
    eList = np.linspace(E_min,E_max,int((E_max-E_min)/deltaE))
    spread = np.zeros((kPts,len(eList)))
    for i in tqdm(range(kPts)):
        for n in range(nCells*minimumBand,nCells*28):
            spread += fs.weight_spreading(weights[i,n],kList[i],evals[i,n],kList,eList[None,:],pars_spread[:3])
    if saveW:
        np.savez(th_w_data_fn,spread=spread,eList=eList)
else:
    print("Loading intensities")
    eList = np.load(th_w_data_fn)['eList']
    spread = np.load(th_w_data_fn)['spread']

""" Plotting """
indK = kPts//2      #SPECIFIC of path G-K-KP !!!!
spGK = spread[:indK,:]
spKKp = spread[indK:,:]

fig = plt.figure(figsize=(20,10))
indKplus = 5   # number of additional points (out of 99) after the K point on the 2 sides)
spKplus = spKKp[:indKplus]
ax1 = fig.add_subplot(121)      #K->G->Kp plot
spKGK = np.concatenate([spKplus[::-1,:],spGK[2:,:][::-1],spGK[2:,:],spKplus],axis=0)
imKGK = spKGK[:,::-1].T
ax1.imshow(
    imKGK,
    cmap='berlin',
)


indKG = 50   # number of additional points (out of 100) after the K point on the 2 sides, towards G
spKplus = spGK[-indKG:-2,]
ax2 = fig.add_subplot(122)      #K->Kp plot
spKKp = np.concatenate([spKplus,spKKp[2:-3],spKplus[::-1,:]],axis=0)
imKKp = spKKp[:,::-1].T
ax2.imshow(
    imKKp,
    cmap='berlin',
)
plt.show()

if input("Save intensity matrix? [y/N]")=='y':
    fnKGK = 'intensityK-G-Kp.csv'
    np.savetxt(fnKGK, imKGK, delimiter=",")
    fnKKp = 'intensityK-Kp.csv'
    np.savetxt(fnKKp, imKKp, delimiter=",")









