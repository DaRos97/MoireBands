"""
Here we take a cut at Gamma to look at the distance between the main band and the side band crossing below it.
We compute, for each set of (phi.w2p,w2d,stacking), the best V which gives the experimental distance:
    - 78 meV for S3
    - 93 meV for S11

We then look also at the full image to see if the parameters make sense.

An extension would be to do the same at K
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

if len(sys.argv) != 3:
    print("Usage: python3 EDC.py arg1 arg2")
    print("arg1: index of arguments")
    print("arg2: S3 or S11")
    quit()
else:
    ind = int(sys.argv[1])
    sample = sys.argv[2]

expVal = {'S3':0.078, 'S11':0.093}
expEDC = expVal[sample]

# Preamble
disp = 1#False
save = True

if disp:
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm

""" Fixed parameetrs """
monolayer_type = 'fit'
Vk,phiK = (0.007,-106/180*np.pi)
nShells = 2
nCells = int(1+3*nShells*(nShells+1))
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees
w1p = cfs.w1p_dic[monolayer_type][sample]
w1d = cfs.w1d_dic[monolayer_type][sample]
kListG = np.array([np.zeros(2),])
if disp and 0:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Interlayer coupling w1: %f, %f"%(w1p,w1d))
    print("Sample ",sample," which has twist ",theta,"°")
    print("Moiré potential at K (%f eV, %f°)"%(Vk,phiK/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)

""" Variable parameters """
nPhi = 25
nW2p = 5
nW2d = 5
listPhi = np.linspace(0,2*np.pi/3,nPhi,endpoint=False)
listW2p = np.linspace(-0.01,0.01,nW2p)
listW2d = np.linspace(-0.01,0.01,nW2d)
stackings = ['P','AP']
stacking,w2p,w2d,phiG = list(itertools.product(*[stackings,listW2p,listW2d,listPhi]))[ind]
if abs(w2p)<1e-8:
    w2p=0
if abs(w2d)<1e-8:
    w2d=0
if abs(phiG)<1e-8:
    phiG=0
parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}
if disp:
    print("---------VARIABLE PARAMETERS CHOSEN---------")
    print("(stacking,w2_p,w2_d,phi) = (%s, %f eV, %f eV, %f°)"%(stacking,w2p,w2d,phiG/np.pi*180))

""" Check we didn't already compute it """
data_fn = 'Data/EDC/Vbest_'+fsm.get_fn(*(sample,nShells))+'.svg'
if Path(data_fn).is_file():
    with open(data_fn,'r') as f:
        l = f.readlines()
        for i in l:
            terms = i.split(',')
            if terms[0] == stacking:
                if terms[1]=="{:.7f}".format(w2p) and terms[2]=="{:.7f}".format(w2d) and terms[3]=="{:.7f}".format(phiG):
                    print("Parameters already computed")
                    quit()

""" Compute distances for many Vs """
listVg = np.linspace(0.001,0.5,40)
distances = np.zeros(len(listVg))
for i in tqdm(range(len(listVg)),desc="Looping Vg"):
    Vg = listVg[i]
    args = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
    disp_fit = False
    distances[i] = fsm.EDC(args,spreadE=0.03,disp=disp_fit)
    if distances[i] == -1:
        distances[i] *= np.nan
        #break

""" Fit to extrct best V """
def poly(x,a,b,c,d,e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4
try:    # Try the fit -> could not work for very random points
    inds = np.array(np.argwhere(abs(distances-expEDC)<0.01))[:,0]
    Vvalues = listVg[inds]
    Dvalues = distances[inds]
    popt, pcov = curve_fit(poly,Vvalues,Dvalues)
    vline = np.linspace(Vvalues[0],Vvalues[-1],100)
    Vbest = vline[np.argmin(abs(poly(vline,*popt)-expEDC))]
    foundVbest = True
except:
    foundVbest = False

""" Plot and save result """
if foundVbest:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(listVg,distances,'r*')
    ax.axhline(expEDC,c='g',lw=2,label=sample)
    ax.axvline(Vbest,c='m',lw=2,label="best V=%f"%Vbest)
    ax.plot(vline,poly(vline,*popt),c='b',ls='--')
    ax.set_xlabel("V")
    ax.set_ylabel("distance")
    ax.set_title(r"$\varphi$, $w_2^p$, $w_2^d$ = %f°, %f, %f"%(phiG/np.pi*180,w2p,w2d))
    ax.legend()
    figname = 'Figures/EDC/DvsV_'+fsm.get_fn(*(stacking,nShells,w2p,w2d,phiG))+'.png'
    fig.savefig(figname)
    if disp:
        print("Best V found")

    """ Save as a line """
    if save:
        with open(data_fn,'a') as file:
            writer = csv.writer(file)
            # Write a single row
            writer.writerow([stacking, "{:.7f}".format(w2p), "{:.7f}".format(w2d), "{:.7f}".format(phiG), Vbest])
























