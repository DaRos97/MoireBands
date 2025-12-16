""" Here we take an intensity cut at K to look at the distance between the main band and the side band crossing below it.
We compute for each phase the best V which gives the experimental position and distance:
    - S3:
    - S11: positions -0.8989 and -1.0696 eV, distance 170 meV (offset of -0.47 eV)

We then look also at the full image to see if the parameters make sense.
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

sample = "S11"
ang_dic = {'-':-0.3, '0':0, '+':0.3}
ang = ang_dic['0']      #Change here for \pm 0.3 degrees
if machine =='maf':
    disp = False
    save = True
else:
    disp = True
    save = False

peak0 = -0. if sample=='S3' else -0.42898
peak1 = -0.7730 if sample=='S3' else -0.5996
peaks = (peak0,peak1)
expVal = {'S3':0.078, 'S11':0.170}
expEDC = expVal[sample]

if disp:
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm

""" Fixed parameetrs """
nShells = 2
monolayer_type = 'fit'
Vg,phiG = (0.017,175/180*np.pi)
nCells = int(1+3*nShells*(nShells+1))
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees, from LEED eperiment
theta += ang
#w1p = cfs.w1p_dic[monolayer_type][sample]
#w1d = cfs.w1d_dic[monolayer_type][sample]
w1p = -1.655
w1d = 0.325
stacking = 'P'
w2p = w2d = 0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
kListK = np.array([
        cfs.R_z(np.pi/3)@np.array([4/3*np.pi/cfs.dic_params_a_mono['WSe2'],0]),     #K
])
spreadE = 0.03      # in eV
if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Sample ",sample," with twist ",theta,"°")
    print("Moiré potential at G (%.5f eV, %.1f°)"%(Vg,phiG/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)
    print("Energy spreading: %.3f eV"%spreadE)

""" Variable parameters """
specificValues = None#(175/180*np.pi,-1.72,0.38)
listVk = np.linspace(0.002,0.009,71)     # Considered values of moirè potential -> every .1 meV
if specificValues is None:
    listPhi = np.linspace(-180/180*np.pi,180/180*np.pi,360,endpoint=False)
else:
    phiK = specificValues
    listPhi = [phiK,]
nPhi = len(listPhi)

""" Computation """
home_dn = fsm.get_home_dn(machine)
data_dn = cfs.getFilename(('kedc',*(sample,nShells,theta)),dirname=home_dn+"Data/kEDC/")+'/'
data_fn = cfs.getFilename(('vbest',*(w1p,w1d,listPhi[0],listPhi[-1],nPhi)),dirname=data_dn,extension='.npy')

bestVs = []
for ip,phiK in enumerate(listPhi):
    if disp:
        print("Considering phase %.1f°"%(phiK/np.pi*180))

    """ Compute peak centers for many Vs """
    centers = np.zeros((len(listVk),2))
    bestInds = []
    for i in tqdm(range(len(listVk)),desc="Looping Vk"):
        Vk = listVk[i]
        args = (nShells, nCells, kListK, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
        fit_disp = False
        fit_plot = False
        a = fsm.EDC(args,sample,peaks,spreadE=spreadE,disp=fit_disp,plot=fit_plot,figname='',show=True)
        if a[0]==-2:    # Weight became larger in lower band -> no coming back from that
            centers[i:,:] = np.nan
            break
        centers[i,0], centers[i,1] = a[0], a[1]

        # We chose a good V if the centers are both within 2.5 meV of the experimental one
        if abs(a[0]-peak0)<0.005 and abs(a[1]-peak1)<0.005:
            if disp:
                print("found best ind at %.6f"%listVk[i])
                #a = fsm.EDC(args,sample,peaks,spreadE=spreadE,disp=True,plot=True,figname='')
            bestInds.append(i)

    if len(bestInds)==0:
        if disp:
            print("No solutions found")
        foundVbest = False
    else:
        foundVbest = True
        bestInd = np.argmin( np.absolute(centers[bestInds,0]-peak0) + np.absolute(centers[bestInds,1]-peak1) )
        Vbest = listVk[bestInds[bestInd]]

    if foundVbest and disp and 0:   #plotting of distances over V
        """ Plot and save result """
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(listVk,centers[:,0]-centers[:,1],'r*')
        ax.axhline(expEDC,c='g',lw=2,label=sample)
        if foundVbest:
            ax.axvline(Vbest,c='m',lw=2,label="best V=%f"%Vbest)
            #ax.plot(vline,poly(vline,*popt),c='b',ls='--')
        ax.set_xlabel("V")
        ax.set_ylabel("distance")
        ax.set_title(r"$\varphi$= %f°"%(phiG/np.pi*180,))
        ax.legend()
        if disp:
            plt.show()
        if save:
            figname = 'Figures/kEDC/DvsV_'+fsm.get_fn(*(sample,nShells,w1p,w1d,phiG))+'.png'
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





















