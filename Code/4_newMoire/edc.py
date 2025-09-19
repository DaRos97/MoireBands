"""
DESCRIPTION:
Here we take a cut at Gamma to look at the distance between the main band and the side band crossing below it.
We compute, for each set of (phi.w2p,w2d,stacking), the best V which gives the experimental distance:
    - 78 meV for S3
    - 93 meV for S11

We then look also at the full image to see if the parameters make sense.

An extension would be to do the same at K
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
from scipy.optimize import curve_fit
import itertools
import csv
machine = cfs.get_machine(os.getcwd())

""" Parameters and options """
parser = argparse.ArgumentParser(description="Calculation of EDC")
parser.add_argument("Index", help="Index of parameters (see list in code)", type=int)
parser.add_argument("Sample", help="Sample to consider (S3 or S11)")
parser.add_argument("Angle", help="Twist angle to use (-,0 or + for -0.3°,0° and +0.3° wrt LEED measured twist)")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
parser.add_argument("-s","--save", help="Save data at the end of calculation", action="store_true")
inputArguments = parser.parse_args()

ind = inputArguments.Index
sample = inputArguments.Sample
ang_dic = {'-':-0.3, '0':0, '+':0.3}
ang = ang_dic[inputArguments.Angle]
disp = inputArguments.verbose
save = inputArguments.save

expVal = {'S3':0.078, 'S11':0.093}
expEDC = expVal[sample]

if disp:
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm

""" Fixed parameetrs """
nShells = 2
monolayer_type = 'fit'
Vk,phiK = (0.007,-106/180*np.pi)
nCells = int(1+3*nShells*(nShells+1))
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees
theta += ang
#w1p = cfs.w1p_dic[monolayer_type][sample]
#w1d = cfs.w1d_dic[monolayer_type][sample]
kListG = np.array([np.zeros(2),])
if disp:    #print what parameters we're using
    print("-----------FIXED PARAMETRS CHOSEN-----------")
    print("Monolayers' tight-binding parameters: ",monolayer_type)
    print("Sample ",sample," which has twist ",theta,"°")
    print("Moiré potential at K (%f eV, %f°)"%(Vk,phiK/np.pi*180))
    print("Number of mini-BZs circles: ",nShells)

""" Variable parameters """
nPhi = 37
nW1p = 21
nW1d = 7
listPhi = np.linspace(0,np.pi,nPhi)
#listPhi = [listPhi[-2],]        #######################
listW1p = np.linspace(-1.68,-1.78,nW1p)
listW1d = np.linspace(0.38,0.44,nW1d)
stackings = ['P',]#['P','AP']
stacking,w1p,w1d = list(itertools.product(*[stackings,listW1p,listW1d]))[ind]
w2p = w2d = 0

parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d}

for phiG in listPhi:
    if disp:
        print("---------VARIABLE PARAMETERS CHOSEN---------")
        print("Interlayer coupling w1: %f, %f"%(w1p,w1d))
        print("(stacking,w2_p,w2_d,phi) = (%s, %.4f eV, %.4f eV, %.1f°)"%(stacking,w2p,w2d,phiG/np.pi*180))
    """ Check we didn't already compute it """
    data_fn = 'Data/EDC/Vbest_'+fsm.get_fn(*(sample,nShells,theta))+'.svg'
    alreadyComputed = False
    if Path(data_fn).is_file():
        with open(data_fn,'r') as f:
            l = f.readlines()
            for i in l:
                terms = i.split(',')
                if terms[0] == stacking:
                    if terms[1]=="{:.7f}".format(w1p) and terms[2]=="{:.7f}".format(w1d) and terms[3]=="{:.7f}".format(phiG):
                        alreadyComputed = True
                        Vbest = float(terms[-1])
                        print("Parameters already computed, Vbest=%.4f eV"%Vbest)
                        break
    """ Computation of best V """
    if not alreadyComputed:
        """ Compute distances for many Vs """
        listVg = np.linspace(0.001,0.05,99)
        distances = np.zeros(len(listVg))
        for i in tqdm(range(len(listVg)),desc="Looping Vg"):
            Vg = listVg[i]
            args = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', False, False)
            disp_fit = 0#False
            disp_plot = 0#False
            distances[i] = fsm.EDC(args,sample,spreadE=0.03,disp=disp_fit,plot=disp_plot,figname='')
            if distances[i] == -2:      #exit loop because the weight moved to the other band so we cannot get a good solution anyway
                break

        """ Extract best V """
        inds = np.argwhere(distances>0)[:,0]
        Vvalues = listVg[inds]
        Dvalues = distances[inds]
        # Check all Vs are in Vvalues within a range
        foundVbest = True
        if len(Vvalues)==0:
            foundVbest = False
        else:
            for iv in range(len(Vvalues)-1):
                if Vvalues[iv+1]-Vvalues[iv] > listVg[1]-listVg[1]:
                    foundVbest=False
        if foundVbest:  #Take middle of interval
            VbestMin = np.min(Vvalues)
            VbestMax = np.max(Vvalues)
            Vbest = (Vvalues[-1]+Vvalues[0])/2
            if disp:
                print("Best V %.4f"%Vbest)

        if 0:       #fitting over V
            def poly(x,a,b,c,d,e):
                return a + b*x + c*x**2 + d*x**3 + e*x**4
            try:    # Try the fit -> could not work for very random points
                inds = np.array(np.argwhere(abs(distances-expEDC)<0.015))[:,0]
                Vvalues = listVg[inds]
                Dvalues = distances[inds]
                if len(Vvalues) < 4 or max(Dvalues)<expEDC or min(Dvalues)>expEDC:
                    if disp:
                        print("Found distances not good enough for the fit")
                    raise ValueError
                popt, pcov = curve_fit(poly,Vvalues,Dvalues)
                vline = np.linspace(Vvalues[0],Vvalues[-1],100)
                Vbest = vline[np.argmin(abs(poly(vline,*popt)-expEDC))]
                foundVbest = True
            except:
                if disp:
                    print("Fit of Vbest didn't work")
                foundVbest = False

        if foundVbest:
            """ Check that the EDC fit works """
            args = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vbest,Vk,phiG,phiK), '', False, False)
            dBest = fsm.EDC(args,sample,spreadE=0.03,disp=disp_fit,plot=disp_plot,figname='')
            if dBest<0:
                if disp:
                    print("Vbest found was a fake -> no fitting in the EDC")
                quit()

            if 0:   #plotting of fit over V
                """ Plot and save result """
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.plot(listVg,distances,'r*')
                ax.axhline(expEDC,c='g',lw=2,label=sample)
                if foundVbest:
                    ax.axvline(Vbest,c='m',lw=2,label="best V=%f"%Vbest)
                    ax.plot(vline,poly(vline,*popt),c='b',ls='--')
                ax.set_xlabel("V")
                ax.set_ylabel("distance")
                ax.set_title(r"$\varphi$, $w_1^p$, $w_1^d$ = %f°, %f, %f"%(phiG/np.pi*180,w1p,w1d))
                ax.legend()
                if disp:
                    plt.show()
                if save:
                    figname = 'Figures/EDC/DvsV_'+fsm.get_fn(*(sample,nShells,w1p,w1d,phiG))+'.png'
                    fig.savefig(figname)

        """ Save as a line """
        if foundVbest and save:
            with open(data_fn,'a') as file:
                writer = csv.writer(file)
                # Write a single row
                writer.writerow([stacking, "{:.7f}".format(w1p), "{:.7f}".format(w1d), "{:.7f}".format(phiG), "{:.7f}".format(VbestMin), "{:.7f}".format(VbestMax), "{:.7f}".format(Vbest)])

    """ Plot image of final result """
    if save:
        figname_final = 'Figures/EDC/final_'+fsm.get_fn(*(sample,nShells,theta,w1p,w1d,phiG))+'.png'
        if 1 or not Path(figname_final).is_file():
            args = (nShells, nCells, kListG, monolayer_type, parsInterlayer, theta, (Vbest,Vk,phiG,phiK), '', False, False)
            fsm.EDC(args,sample,spreadE=0.03,disp=True,plot=True,figname=figname_final)






















