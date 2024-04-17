import numpy as np
import functions as fs
import parameters as ps
import os

from contextlib import redirect_stdout

"""
We check the orbital content of the bands around Gamma and K for the solutions found with the minimization.
Consider only TVB and lower 4 bands.
We want to find something close to the values of DFT
"""

machine = 'loc'
final_fit_dn = 'temp/' #or ''

orbital_fn = fs.get_home_dn('loc') + 'results/orbital_content.txt'
with open(orbital_fn, 'w') as f:
    with redirect_stdout(f):
        print(" Orbital content of minimization solutions: \n\n##########################################\n")
        for ind in range(40):
            TMD,fixed_SO,range_par = fs.get_parameters(ind)
            print("Computing TMD: ",TMD,", fixed SO: ",fixed_SO," and range: ",range_par)

            pars = [0,]
            for file in os.listdir(fs.get_home_dn(machine)+'results/'+final_fit_dn):
                terms = file.split('_')
                if terms[1] == TMD and terms[2]=="{:.2f}".format(range_par).replace('.',',') and str(fixed_SO)==terms[3]:
                    pars = np.load(fs.get_home_dn(machine)+'results/'+final_fit_dn+file)
                    chi2 = terms[-1][:-4]
                    break
            if len(pars)==1:
                print("Parameters not found for TMD: ",TMD,", range_par: ",range_par," and SO fixed: ",fixed_SO,'\n')
                continue
            
            a_mono = ps.dic_params_a_mono[TMD]
            k_pts = [np.array([0,0]),np.array([4/3*np.pi/a_mono,0])]
            if fixed_SO:
                SO_values = ps.initial_pt[TMD][-2:]
                full_pars = list(pars)
                for i in range(2):
                    full_pars.append(SO_values[i])
            else:
                full_pars = pars
            
            for k in k_pts:
                H = fs.H_monolayer(k,fs.find_t(full_pars),fs.find_e(full_pars),fs.find_HSO(full_pars),a_mono,pars[-3])
                E,evec = np.linalg.eigh(H)
                if (k == np.array([0,0])).all():
                    dz2 = np.sqrt(np.linalg.norm(evec[5,13])**2+np.linalg.norm(evec[16,13])**2)
                    pze = np.sqrt(np.linalg.norm(evec[8,13])**2+np.linalg.norm(evec[19,13])**2)
                    print("Gamma: ")
                    print("d_{z^2}: ",dz2)
                    print("p_{z}^e: ",pze)
                    print("weight: ",dz2**2+pze**2)
                else:
                    d2 = np.sqrt(np.linalg.norm(evec[6,13])**2+np.linalg.norm(evec[7,13])**2+np.sqrt(np.linalg.norm(evec[17,13])**2+np.linalg.norm(evec[18,13])**2))
                    p1e = np.sqrt(np.linalg.norm(evec[9,13])**2+np.linalg.norm(evec[10,13])**2+np.sqrt(np.linalg.norm(evec[20,13])**2+np.linalg.norm(evec[21,13])**2))
                    print("K: ")
                    print("d_{2}: ",d2)
                    print("p_{1}^e: ",p1e)
                    print("weight: ",d2**2+p1e**2)
            print('____________________________________________\n')

