import numpy as np
import functions as fs
import parameters as ps
import os

"""
We check the orbital content of the bands around Gamma and K for the solutions found with the minimization.
Consider only TVB and lower 4 bands.
We want to find something close to the values of DFT
"""

machine = 'loc'
final_fit_dn = 'temp/' #or ''

for ind in range(40):
    TMD,considered_cuts,range_par = fs.get_parameters(ind)
    cuts_fn = ''
    for i in range(len(considered_cuts)):
        cuts_fn += considered_cuts[i]
        if i != len(considered_cuts)-1:
            cuts_fn += '_'
    print("Computing TMD: ",TMD,", in cuts: ",cuts_fn," and range: ",range_par)


    for file in os.listdir(fs.get_home_dn(machine)+'results/'+final_fit_dn):
        if file[5:5+len(TMD)] == TMD and file[6+len(TMD):10+len(TMD)]=="{:.2f}".format(range_par).replace('.',',') and file[11+len(TMD):11+len(TMD)+len(cuts_fn)]==cuts_fn:
            pars = np.load(fs.get_home_dn(machine)+'results/'+final_fit_dn+file)
            chi2 = file[-10:-4]
    
    a_mono = ps.dic_params_a_mono[TMD]
    k_pts = [np.array([0,0]),np.array([4/3*np.pi/a_mono,0])]
    
    for k in k_pts:
        H = fs.H_monolayer(k,fs.find_t(pars),fs.find_e(pars),fs.find_HSO(pars),a_mono,pars[-1])
        E,evec = np.linalg.eigh(H)
        if (k == np.array([0,0])).all():
            dz2 = np.sqrt(np.linalg.norm(evec[5,13])**2+np.linalg.norm(evec[16,13])**2)
            pze = np.sqrt(np.linalg.norm(evec[8,13])**2+np.linalg.norm(evec[19,13])**2)
            print("Gamma: ")
            print("d_{z^2}: ",dz2)
            print("p_{z}^e: ",pze)
            print("weight: ",dz2**2+pze**2,'\n')
        else:
            d2 = np.sqrt(np.linalg.norm(evec[6,13])**2+np.linalg.norm(evec[7,13])**2+np.sqrt(np.linalg.norm(evec[17,13])**2+np.linalg.norm(evec[18,13])**2))
            p1e = np.sqrt(np.linalg.norm(evec[9,13])**2+np.linalg.norm(evec[10,13])**2+np.sqrt(np.linalg.norm(evec[20,13])**2+np.linalg.norm(evec[21,13])**2))
            print("K: ")
            print("d_{2}: ",d2)
            print("p_{1}^e: ",p1e)
            print("weight: ",d2**2+p1e**2,'\n')

