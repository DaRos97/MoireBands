import sys,os
import numpy as np
cwd = os.getcwd()
master_folder = cwd[:40]
sys.path.insert(1, master_folder)
import CORE_functions as cfs
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils


fnWSe2 = '../Inputs/tb_WSe2_B:5_K:0.00005_0.1_0_1_0.1_10.npy'
fnWS2 = '../Inputs/tb_WS2_B:5_K:0.0001_0.1_0_1_0.1_5.npy'

""" Save energy and intensities """
saveE = True
saveW = True

""" Bilayer parameters -> changing any of these you need to re-evaluate the energies """
kPlusKGKp = 0.2
kPlusKMKp = 0.6
kDelta = 5e-3
theta = 2.8     # twisting angle, in deg
Vk = 0.007              # eV
phiK = 106/180*np.pi       # rad
Vg = 0.020              # eV
phiG = 173/180*np.pi        # rad
w1p = -0.47     # eV
w1d = 1.03      # eV
#Vg = 0.014              # eV
#phiG = 169/180*np.pi        # rad
#w1p = -1.1      # eV
#w1d = 0.42       # eV

print(Vg,phiG,w1p,w1d)

""" Parameters of intensity matrix -> changing any of these you need to re-evaluate the intensities """
typeSpread = 'Gauss'    # 'Gauss' or 'Lorentz', works for both k and E
spreadK = 0.01     # in 1/a
spreadE = 0.015      # in eV
E_max = 0           # in eV
E_min = -2.5        # in eV
deltaE = 0.01       # in eV, sets the energy grid
shadeFactorWS2 = 0.0     # shade factor of WS2 -> 0 (not visible at all) to 1 (same relevance as WSe2)
powFactor = 1.       # exponent of weights -> 2 is the usual mod square of eigenvectors which should be the weight of ARPES spectra. For lower values we enhance the intensity of the side bands
minimumBand = 15    # lowest considered band for spreading      #bands are 0 to 43, with TVB at 27

""" Parameters of final plot """
shadeFactorE = 0.2 # Add a linar shading depending on the energy. Starts at 1 at E_max and goes to this factor at E_min

""" Actual computation """

""" Interlayer parameters """
sample = 'S11'
stacking = 'P'
w2p = w2d = 0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
nShells = 2
nCells = int(1+3*nShells*(nShells+1))
monolayer_fns = {'WSe2':fnWSe2,'WS2':fnWS2}
moire_pars = (Vg,Vk,phiG,phiK)
""" BZ path and energy list """
kListKGKp = cfs.getMomentaKGKp(kPlusKGKp,kDelta)
normKGKp = kListKGKp[:,0]
kPtsKGKp = kListKGKp.shape[0]
kListKMKp, normKMKp = cfs.getMomentaKMKp(kPlusKMKp,kDelta)
kPtsKMKp = kListKMKp.shape[0]
modK = 4*np.pi/3/cfs.dic_params_a_mono['WSe2']
eList = np.linspace(E_min,E_max,int((E_max-E_min)/deltaE))
""" Computing evals and evecs """
args_e_data = (sample,nShells,Vg,phiG,Vk,phiK,theta,w1p,w1d,kPlusKGKp,kPlusKMKp,kDelta)
th_e_data_fn = cfs.getFilename(('figMoire_e',)+args_e_data,dirname='Data/',extension='.npz')
if not Path(th_e_data_fn).is_file():
    print("Computing energies KGKp")
    args = (nShells, nCells, kListKGKp, monolayer_fns, parsInterlayer, theta, moire_pars, '', False, True)
    evalsFullKGKp, evecsFullKGKp = utils.diagonalize_matrix(*args)
    evalsKGKp = evalsFullKGKp[:,nCells*minimumBand:nCells*28]
    evecsKGKp = evecsFullKGKp[:,:,nCells*minimumBand:nCells*28]
    print("Computing energies KMKp")
    args = (nShells, nCells, kListKMKp, monolayer_fns, parsInterlayer, theta, moire_pars, '', False, True)
    evalsFullKMKp, evecsFullKMKp = utils.diagonalize_matrix(*args)
    evalsKMKp = evalsFullKMKp[:,nCells*minimumBand:nCells*28]
    evecsKMKp = evecsFullKMKp[:,:,nCells*minimumBand:nCells*28]
    if saveE:
        np.savez(
            th_e_data_fn,
            evalsKGKp=evalsKGKp,
            evecsKGKp=evecsKGKp,
            evalsKMKp=evalsKMKp,
            evecsKMKp=evecsKMKp,
        )
else:
    print("Loading energies")
    evalsKGKp = np.load(th_e_data_fn)['evalsKGKp']
    evecsKGKp = np.load(th_e_data_fn)['evecsKGKp']
    evalsKMKp = np.load(th_e_data_fn)['evalsKMKp']
    evecsKMKp = np.load(th_e_data_fn)['evecsKMKp']
""" Computing weights and spread image """
pars_spread = ( spreadK, spreadE, typeSpread, deltaE, powFactor, shadeFactorWS2, minimumBand)
args_w_data = args_e_data + pars_spread
th_w_data_fn = cfs.getFilename(('figMoire_w',)+args_w_data,dirname='Data/',extension='.npz')
if not Path(th_w_data_fn).is_file():
    print("Computing intensities KGKp")
    weightsKGKp = np.zeros((kPtsKGKp,nCells*(28-minimumBand)))
    for i in range(kPtsKGKp):
        ab = np.absolute(evecsKGKp[i])**powFactor
        weightsKGKp[i,:] = np.sum(ab[:22,:],axis=0) + shadeFactorWS2*np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    spreadKGKp = np.zeros((kPtsKGKp,len(eList)))
    for i in tqdm(range(kPtsKGKp)):
        for n in range(nCells*(28-minimumBand)):#,nCells*28):
            spreadKGKp += utils.weight_spreading(
                weightsKGKp[i,n],
                kListKGKp[i],
                evalsKGKp[i,n],
                kListKGKp,
                eList[None,:],
                pars_spread[:3])
    print("Computing intensities KMKp")
    weightsKMKp = np.zeros((kPtsKMKp,nCells*(28-minimumBand)))
    for i in range(kPtsKMKp):
        ab = np.absolute(evecsKMKp[i])**powFactor
        weightsKMKp[i,:] = np.sum(ab[:22,:],axis=0) + shadeFactorWS2*np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    spreadKMKp = np.zeros((kPtsKMKp,len(eList)))
    for i in tqdm(range(kPtsKMKp)):
        for n in range(nCells*(28-minimumBand)):#,nCells*28):
            spreadKMKp += utils.weight_spreading(
                weightsKMKp[i,n],
                kListKMKp[i],
                evalsKMKp[i,n],
                kListKMKp,
                eList[None,:],
                pars_spread[:3])
    if saveW:
        np.savez(
            th_w_data_fn,
            spreadKGKp=spreadKGKp,
            spreadKMKp=spreadKMKp,
        )
else:
    print("Loading intensities")
    spreadKGKp = np.load(th_w_data_fn)['spreadKGKp']
    spreadKMKp = np.load(th_w_data_fn)['spreadKMKp']

""" Plotting """
fig = plt.figure(figsize=(18,6))
ax1 = fig.add_subplot(121)      # K->G->Kp plot
ax1.pcolormesh(
    normKGKp,
    eList,
    spreadKGKp.T,
    cmap='Greys',
    shading="auto"
)

ax2 = fig.add_subplot(122)      # K->Kp plot
ax2.pcolormesh(
    normKMKp,
    eList,
    spreadKMKp.T,
    cmap='Greys',
    shading='auto'
)

fig.tight_layout()
plt.show()

""" Run this part to save as .csv file the last plotted intensities and the momentum/energy values"""
if 0:
    fnKGK = 'Data/intensity_K-G-Kp.csv'
    np.savetxt(fnKGK, spKGK, delimiter=",")
    fnKGKm = 'Data/momenta_K-G-Kp.csv'
    np.savetxt(fnKGKm, K1, delimiter=",")
    fnKGKe = 'Data/energy_K-G-Kp.csv'
    np.savetxt(fnKGKe, eList, delimiter=",")

    fnKK = 'Data/intensity_K-Kp.csv'
    np.savetxt(fnKK, spKKp, delimiter=",")
    fnKKm = 'Data/momenta_K-Kp.csv'
    np.savetxt(fnKKm, K2, delimiter=",")
    fnKKe = 'Data/energy_K-Kp.csv'
    np.savetxt(fnKKe, eList, delimiter=",")












































