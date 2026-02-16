import sys,os
import numpy as np
cwd = os.getcwd()
master_folder = cwd[:40]
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions as fs
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

save = True

fig,axs = plt.subplots(
    2,2,
    width_ratios=[1,0.5],
    figsize=(7,4),
)     #PRB is ~ 7x9.5 inches

""" ARPES intensity """
TMD = 'WSe2'
ARPESfn = 'Inputs/intensity_GKM_%s_sum_BE_crop.txt'%TMD
ARPES_intensity = np.loadtxt(ARPESfn, delimiter="\t")

nK,nE = ARPES_intensity.shape
listE = np.linspace(-3.5,0,nE)
kEnd = 1.9896 if TMD=='WS2' else 1.9075
listK = np.linspace(0,kEnd,nK,endpoint=True)

""" Extracted data ARPES """
dataObject = cfs.dataWS2() if TMD=="WS2" else cfs.dataWSe2()
ptsPerPath = (40,20,20)
ARPES_energy = dataObject.getFitData(ptsPerPath)


""" Fit and DFT bands. """
args_e_data = (TMD,nK)
th_e_data_fn = cfs.getFilename(('en',)+args_e_data,dirname='Data/',extension='.npz')
if not Path(th_e_data_fn).is_file():
    print("Computing monolayer energy")
    DFT_pars = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC
    HSO_DFT = cfs.find_HSO(DFT_pars[-2:])
    evals_dft = cfs.energy(DFT_pars,HSO_DFT,ARPES_energy,TMD,bands=np.arange(22))

    fit_pars = np.load("Inputs/tb_%s.npy"%TMD)
    HSO_fit = cfs.find_HSO(fit_pars[-2:])
    evals_fit = cfs.energy(fit_pars,HSO_fit,ARPES_energy,TMD,bands=np.arange(22))
    if save:
        np.savez(th_e_data_fn,fit=evals_fit,dft=evals_dft)
else:
    print("Loading evals")
    evals_fit = np.load(th_e_data_fn)['fit']
    evals_dft = np.load(th_e_data_fn)['dft']

#kList,norm = cfs.get_kList('G-K-M',nK,TMD=TMD,endpoint=True,returnNorm=True)
# norm is similar but slightly different than listK of ARPES
""" Intensity matrix of fit. """
typeSpread = 'Gauss'    # 'Gauss' or 'Lorentz', works for both k and E
spreadK = 0.005     # in 1/a
spreadE = 0.03      # in eV
pars_spread = ( spreadK, spreadE, typeSpread)
args_w_data = args_e_data + pars_spread
th_w_data_fn = cfs.getFilename(('we',)+args_w_data,dirname='Data/',extension='.npy')
if not Path(th_w_data_fn).is_file():
    print("Computing intensities")
    weights = np.ones((nK,44))
    spread_fit = np.zeros((nK,nE))
    for i in tqdm(range(nK)):
        for n in range(44):
            spread_fit += fs.weight_spreading(weights[i,n],kList[i],evals_fit[i,n],kList,listE[None,:],pars_spread[:3])
    if save:
        np.save(th_w_data_fn,spread_fit)
else:
    print("Loading intensities")
    spread_fit = np.load(th_w_data_fn)

print(spread_fit.shape)
print(data.shape)

data /= np.max(data)
spread_fit /= np.max(spread_fit)

listK_full = np.concatenate([listK,norm[::-1]])
spread_full = np.concatenate([data,spread_fit],axis=0)

ax = axs[0,0]
ax.pcolormesh(
    listK,
    listE,
    (data**0.5).T,
    cmap='gray_r'
)
ax = axs[1,0]
ax.pcolormesh(
    norm,
    listE,
    spread_fit.T,
    cmap='gray_r'
)
ax = axs[1,1]
ymin,ymax = ax.get_ylim()
print(evals_fit.shape)
for i in range(44):
    ax.plot(
        norm,
        evals_fit[:,i],
        color='r'
    )

ax = axs[0,1]
ax.scatter(
    kList[:,0],
    kList[:,1]
)
ax.set_aspect('equal')


plt.show()
exit()


ax = axs[1,0]
ax.pcolormesh(
    listK_full,
    listE,
    spread_full.T,
    cmap='gray_r'
)

plt.show()
