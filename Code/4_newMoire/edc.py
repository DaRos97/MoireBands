""" Here we take an intensity cut at Gamma to look at the distance between the main band and the side band crossing below it.
We compute, for each set of (phi,w1p,w1d), the best V which gives the experimental position and distance:
    - S3:  positions -0.6948 and -0.7730 eV, distance 78 meV
    - S11: positions -1.1599 and -1.2531 eV, distance 93 meV (offset of -0.47 eV)

We then look also at the full image to see if the parameters make sense.

An extension would be to do the same at K.
"""

import sys,os
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
from scipy.optimize import curve_fit
import itertools
import csv
machine = cfs.get_machine(os.getcwd())

""" Parameters and options """
if len(sys.argv)!=2:
    print("Usage: python3 edc.py arg1")
    print("arg1 is an int for the index of the parameter list")
    exit()

ind = int(sys.argv[1])
sample = "S11"
ang_dic = {'-':-0.3, '0':0, '+':0.3}
ang = ang_dic['0']      #Change here for \pm 0.3 degrees
if machine =='maf':
    disp = False
    save = True
else:
    disp = True
    save = False

peak0 = -0.6948 if sample=='S3' else -0.6899
peak1 = -0.7730 if sample=='S3' else -0.7831
#expVal = {'S3':0.078, 'S11':0.093}
#expEDC = expVal[sample]

if disp:
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm

""" Fixed parameetrs """
nShells = 2
monolayer_type = 'fit'
Vk,phiK = (0.007,-106/180*np.pi)
nCells = int(1+3*nShells*(nShells+1))
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees, from LEED eperiment
theta += ang
#w1p = cfs.w1p_dic[monolayer_type][sample]
#w1d = cfs.w1d_dic[monolayer_type][sample]
kListG = np.array([np.zeros(2),])
spreadE = 0.03      # in eV
if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Sample ",sample," with twist ",theta,"°")
    print("Moiré potential at K (%.5f eV, %.1f°)"%(Vk,phiK/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)
    print("Energy spreading: %.3f eV"%spreadE)

""" Variable parameters """
listVg = np.linspace(0.001,0.05,99)     # Considered values of moirè potential
listPhi = np.linspace(0,2*np.pi,72,endpoint=False)
nPhi = len(listPhi)
listW1p = np.linspace(-1.625,-1.825,41)         # Values to change ##################################################
listW1d = np.linspace(0.27,0.47,41)           # values to change ##################################################
stacking = 'P'
w1p,w1d = list(itertools.product(*[listW1p,listW1d]))[ind]
w2p = w2d = 0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
if disp:
    print("---------VARIABLE PARAMETERS CHOSEN---------")
    print("Interlayer coupling w1: %.5f, %.5f"%(w1p,w1d))
    print("(stacking,w2_p,w2_d) = (%s, %.4f eV, %.4f eV)"%(stacking,w2p,w2d))

""" Computation """
home_dn = fsm.get_home_dn(machine)
data_dn = cfs.getFilename(('edc',*(sample,nShells,theta)),dirname=home_dn+"Data/newEDC/")+'/'
data_fn = cfs.getFilename(('vbest',*(w1p,w1d,listPhi[0],listPhi[-1],nPhi)),dirname=data_dn,extension='.npy')
if Path(data_fn).is_file():
    print("Already computed set of w1p and w1d for the same phases")
    exit()

bestVs = []
for ip,phiG in enumerate(listPhi):
    if disp:
        print("Considering phase %.1f°"%(phiG/np.pi*180))

    """ Compute peak centers for many Vs """
    centers = np.zeros((len(listVg),2))
    bestInds = []
    for i in tqdm(range(len(listVg)),desc="Looping Vg"):
        Vg = listVg[i]
        args = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
        fit_disp = 0#False
        fit_plot = 0#False
        a = fsm.EDC(args,sample,spreadE=spreadE,disp=fit_disp,plot=fit_plot,figname='')
        if a[0]==-2:    # Weight became larger in lower band -> no coming back from that
            centers[i:,:] = np.nan
            break
        centers[i,0], centers[i,1] = a[0], a[1]

        # We chose a good V if the centers are both within 2.5 meV of the experimental one
        if abs(a[0]-peak0)<0.0025 and abs(a[1]-peak1)<0.0025:
            bestInds.append(i)

    if len(bestInds)==0:
        foundVbest = False
    else:
        foundVbest = True
        bestInd = np.argmin( np.absolute(centers[bestInds,0]-peak0) + np.absolute(centers[bestInds,1]-peak1) )
        Vbest = listVg[bestInds[bestInd]]

    if 0:   #plotting of distances over V
        """ Plot and save result """
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(listVg,centers[:,0]-centers[:,1],'r*')
        ax.axhline(expEDC,c='g',lw=2,label=sample)
        if foundVbest:
            ax.axvline(Vbest,c='m',lw=2,label="best V=%f"%Vbest)
            #ax.plot(vline,poly(vline,*popt),c='b',ls='--')
        ax.set_xlabel("V")
        ax.set_ylabel("distance")
        ax.set_title(r"$\varphi$, $w_1^p$, $w_1^d$ = %f°, %f, %f"%(phiG/np.pi*180,w1p,w1d))
        ax.legend()
        if disp:
            plt.show()
        if save:
            figname = 'Figures/EDC/DvsV_'+fsm.get_fn(*(sample,nShells,w1p,w1d,phiG))+'.png'
            fig.savefig(figname)
        exit()

    if foundVbest:
        bestVs.append((ip, Vbest))

if save:
    if not Path(data_dn).is_dir():
        print("Creating folder "+data_dn)
        os.system("mkdir "+data_dn)
    data = np.array(bestVs)
    np.save(
        data_fn,
        data
    )





















