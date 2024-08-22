import numpy as np
import functions as fs
import parameters as ps
import os

from contextlib import redirect_stdout

"""
We check the orbital content of the bands around Gamma and K for the solutions found with the minimization.
Consider only TVB and lower 4 bands.
We want to find something close to the values of DFT.
"""

machine = 'loc'
type_bound = 'large'
TMD = fs.TMDs[0]
a_mono = ps.dic_params_a_mono[TMD]
k_pts = np.array([np.zeros(2),np.matmul(fs.R_z(np.pi/3),np.array([4/3*np.pi/a_mono,0]))])    #Gamma and K 

print("Computing TMD: ",TMD)

file = fs.get_res_dn(machine)+'result_'+TMD+'.npy'
full_pars = np.load(file)
DFT_pars = np.array(ps.initial_pt[TMD])

args_DFT = (fs.find_t(DFT_pars),fs.find_e(DFT_pars),fs.find_HSO(DFT_pars[-2:]),a_mono,DFT_pars[-3])
H_all = fs.H_monolayer(k_pts,*args_DFT)

print("DFT values:")
for i in range(2):
    H = H_all[i,11:,11:]    #Spinless Hamiltonian
    k = k_pts[i]
    E,evec = np.linalg.eigh(H)
    if 0:   #Print all non 0
        print()
        for b in range(11):
            print("BAND ",b+1,'\n')
            print("Energy: ",E[b])
            a = evec[:,b]
            print("Orbitals:")
            for n in range(11):
                if np.absolute(a[n])>1e-4:
                    print(fs.orb_txt[n],':\t',a[n])
            print("\nMagnetization orbitals:")
            for n in range(11):
                v = fs.fun[n](a)
                if np.absolute(v)>1e-4:
                    print(fs.txt[n],'\t',v)
                    print("abs:\t",np.abs(v))
                    print()
            print("____________________________________________________")
            input()

    a = evec[:,6]
    if (k == np.array([0,0])).all():
        print("Gamma: ")
        d0 = np.absolute(fs.d0(a))
        p0e = np.absolute(fs.p0e(a))
        print("d0e: ",d0)
        print("p0e: ",p0e)
        print("weight: ",d0**2+p0e**2)
    else:
        print("K: ")
        dp2 = np.absolute(fs.dp2(a))
        ppe = np.absolute(fs.ppe(a))
        print("dp2: ",dp2)
        print("ppe: ",ppe)
        print("weight: ",dp2**2+ppe**2)
print('____________________________________________\n')
print('____________________________________________\n')
print("Result values:")
args_res = (fs.find_t(full_pars),fs.find_e(full_pars),fs.find_HSO(full_pars[-2:]),a_mono,full_pars[-3])
H_all = fs.H_monolayer(k_pts,*args_res)
for i in range(2):
    H = H_all[i,11:,11:]    #Spinless Hamiltonian
    k = k_pts[i]
    E,evec = np.linalg.eigh(H)
    a = evec[:,6]
    if (k == np.array([0,0])).all():
        print("Gamma: ")
        d0 = np.absolute(fs.d0(a))
        p0e = np.absolute(fs.p0e(a))
        print("d0e: ",d0)
        print("p0e: ",p0e)
        print("weight: ",d0**2+p0e**2)
    else:
        print("K: ")
        dp2 = np.absolute(fs.dp2(a))
        ppe = np.absolute(fs.ppe(a))
        print("dp2: ",dp2)
        print("ppe: ",ppe)
        print("weight: ",dp2**2+ppe**2)

