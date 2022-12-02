import functions
import numpy as np
import sys
import getopt
import matplotlib.pyplot as plt

def H_tot(k,t,V):
    H = np.zeros((4,4),dtype=complex)
    for i in range(4):
        k_ = k+i*np.pi/2
        H[i,i] = -2*t*np.cos(k_)
    H[0,1] = V
    H[1,2] = V
    H[2,3] = V
    H[0,3] = V
    H[1,0] = V
    H[2,1] = V
    H[3,2] = V
    H[3,0] = V
    return H


pts = 1000
ks = np.linspace(0,np.pi,pts)
weight = np.zeros((pts,4))
en = np.zeros((pts,4))
for i,k in enumerate(ks):
    evals,evecs = np.linalg.eigh(H_tot(k,1,0.1))
    en[i] = evals
    for e in range(len(evals)):
        weight[i,e] = np.abs(evecs[:,e][0])**2


plt.figure()
for i in range(4):
    plt.plot(ks,en[:,i],'k-',lw=0.1)
for i in range(pts):
    for j in range(4):
        plt.scatter(ks[i],en[i,j],color='r',marker='.',s=weight[i,j]**2)
plt.show()

