"""
Here we plot the final image, given some spread and other parameters.
On the left the experimental image and on the right the theoretical result.
"""

import sys,os
import argparse
import numpy as np
import scipy
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_moire as fsm
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
machine = cfs.get_machine(os.getcwd())


""" Parameters and options """
parser = argparse.ArgumentParser(description="Plotting of final image")
parser.add_argument("Sample", help="Sample to consider (S3 or S11)")
parser.add_argument("powInd", help="Power to which elevate weights in theoretical image (1 for right weights, 0.5 for enhanced side bands)", type=float)
parser.add_argument("spread_E", help="Spreading in energy (eV) -> 0.03 is 30 meV", type=float)
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()

sample = inputArguments.Sample
powInd = inputArguments.powInd
spread_E = inputArguments.spread_E
disp = inputArguments.verbose

""" Fixed parameters of theoretical image """
nShells = 2
monolayer_type = 'fit'
Vk,phiK = (0.007,-106/180*np.pi)
nCells = int(1+3*nShells*(nShells+1))
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees
w1p = -1.7 if sample=='S3' else -1.73       #interlayer
w1d = 0.38 if sample=='S3' else 0.39
stacking = 'P'
w2p=w2d = 0
phiG = np.pi/3
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Interlayer coupling w1: %f, %f"%(w1p,w1d))
    print("Sample ",sample," which has twist ",theta,"°")
    print("Moiré potential at K (%f eV, %f°)"%(Vk,phiK/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)
    print("(stacking,w2_p,w2_d,phi) = (%s, %.4f eV, %.4f eV, %.1f°)"%(stacking,w2p,w2d,phiG/np.pi*180))

""" Import best V from EDC fitting """
Vg = -1
data_fn = 'Data/EDC/Vbest_'+fsm.get_fn(*(sample,nShells,theta))+'.svg'
if Path(data_fn).is_file():
    with open(data_fn,'r') as f:
        l = f.readlines()
        for i in l:
            terms = i.split(',')
            if terms[0]==stacking and terms[1]=="{:.7f}".format(w1p) and terms[2]=="{:.7f}".format(w1d) and terms[3]=="{:.7f}".format(phiG):
                Vg = float(terms[-1])
                break
else:
    print("Data file not found: ",data_fn)
    quit()
if Vg==-1:
    print("Values of stacking,w1p,w1d and phiG not found in fit: %s, %.3f, %.3f, %.1f"%(stacking,w1p,w1d,phiG/np.pi*180))
    quit()

""" Left half experimental image"""
from PIL import Image
E_max, E_min, pKi, pKf, pEmax, pEmin = cfs.dic_pars_samples[sample]
fig_fn = fsm.get_inputs_dn(machine) + sample + '_KGK.png'
pic_raw = np.array(np.asarray(Image.open(fig_fn)))
totPe,totPk,_ = pic_raw.shape
kList = cfs.get_kList('Kp-G-K',11)
K0 = np.linalg.norm(kList[0])   #val of |K|
pK0 = int((pKf+pKi)/2)   #pixel of middle -> k=0
pKF = int((pKf-pK0)*K0+pK0)   #pixel of k=|K|
pKI = 2*pK0-pKF             #pixel of k=-|K|
exp_pic = pic_raw[pEmax:pEmin,pKI:pKF]

""" Right half theoretical image """
kPts = 200
kList = np.zeros((kPts,2))
kList[:,0] = np.linspace(0,K0,kPts)
""" Evals and weights """
args_e_data = (sample,nShells,monolayer_type,Vk,phiK,theta,stacking,w2p,w2d,phiG,kPts)
th_e_data_fn = 'Data/final_data_e_'+fsm.get_fn(*args_e_data)+'.npz'
if not Path(th_e_data_fn).is_file():
    moire_pars = (Vg,Vk,phiG,phiK)
    args = (nShells, nCells, kList, monolayer_type, parsInterlayer, theta, moire_pars, '', False, True)
    evals, evecs = fsm.diagonalize_matrix(*args)
    weights = np.zeros((kPts,nCells*44))
    for i in range(kPts):
        ab = np.absolute(evecs[i])**2
        weights[i,:] = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)
    np.savez(th_e_data_fn,evals=evals,evecs=evecs,weights=weights)
else:
    evals = np.load(th_e_data_fn)['evals']
#    evecs = np.load(th_e_data_fn)['evecs']
    weights = np.load(th_e_data_fn)['weights']
if sample=='S11':
    evals -= 0.47

""" Spread """
spread_K = 0.001
spread_type = 'Lorentz'
#spread_type = 'Gauss'
Epts = exp_pic.shape[0]
args_s_data = (sample,nShells,monolayer_type,Vk,phiK,theta,stacking,w2p,w2d,phiG,spread_E,spread_K,spread_type,kPts)
th_s_data_fn = 'Data/final_data_s_'+fsm.get_fn(*args_s_data)+'.npy'
if not Path(th_s_data_fn).is_file():
    E_list = np.linspace(E_min,E_max,Epts)
    spread = np.zeros((kPts,Epts))
    pars_spread = (spread_K,spread_E,spread_type)
    indexMainBand = 28*nCells - 1
    for i in tqdm(range(kPts),desc='Spreading'):
        for n in range(indexMainBand-nCells*10+1,indexMainBand+1):
            spread += fsm.weight_spreading(weights[i,n],kList[i],evals[i,n],kList,E_list[None,:],pars_spread)
    np.save(th_s_data_fn,spread)
else:
    spread = np.load(th_s_data_fn)

""" Figure """
# Truncate energy window
eminInd = 1150 if sample == 'S11' else 1100
emaxInd = 300 if sample == 'S11' else 80   #150
spread = spread.T[::-1,:]
spread = spread[emaxInd:eminInd]
exp_pic = exp_pic[emaxInd:eminInd]
# Normalize
the_pic = (spread/np.max(spread) )**powInd
exp_pic = (255-exp_pic[:,:,0])/255
exp_pic = (exp_pic + exp_pic[:,::-1] )/2
exp_pic = exp_pic[:,:exp_pic.shape[1]//2]

# Resize pics to put them in the same plot
from skimage.transform import resize
w1 = 100
w2 = 100
A_resized = resize(exp_pic, (Epts, w1), preserve_range=True, anti_aliasing=True)
A_resized /= np.max(A_resized)
B_resized = resize(the_pic, (Epts, w2), preserve_range=True, anti_aliasing=True)
B_resized /= np.max(B_resized[emaxInd:eminInd])
combined = np.hstack([A_resized, B_resized]) + 1

from matplotlib.colors import LogNorm
norm = LogNorm(vmin=np.min(combined),vmax=np.max(combined))

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot()
ax.imshow(
    combined,
    cmap='viridis',
#    cmap='plasma',
    aspect='auto',
    norm=norm,
)

ax.text(0.2,0.8,"Experiment",transform=ax.transAxes,color='lawngreen',fontsize=20)
ax.text(0.7,0.8,"Theory",transform=ax.transAxes,color='lawngreen',fontsize=20)
ax.set_title("Sample %s"%sample[1:],size=30)
#ax.set_ylim(eminInd,emaxInd)

ax.axis('off')

plt.tight_layout()
plt.show()

if input("Save?[y/N]")=='y':
    args_fig = args_s_data + (powInd,)
    figname = 'Figures/final_'+fsm.get_fn(*args_fig)+'.png'
    fig.savefig(figname)


































