import functions
import numpy as np
import sys
import getopt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#Test code for trying out the arpes weights


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


pts = 500
ks = np.linspace(0,np.pi,pts)
weight = np.zeros((pts,4))
en = np.zeros((pts,4))
for i,k in enumerate(ks):
    evals,evecs = np.linalg.eigh(H_tot(k,1,0.1))
    en[i] = evals
    for e in range(len(evals)):
        weight[i,e] = np.abs(evecs[:,e][0])**2

plt.figure()
plt.subplot(1,2,1)
for i in range(4):
    plt.plot(ks,en[:,i],'k-',lw=0.1)
for i in range(pts):
    for j in range(4):
        if weight[i,j] > 0.1:
            plt.scatter(ks[i],en[i,j],color='r',marker='.',s=weight[i,j]**2)

plt.subplot(1,2,2)
K_ = 0.004
E_ = 0.4
xd = 100
yd = 100
lor = np.zeros((xd,yd))
KKs = np.linspace(0,np.pi,xd)
YY = np.linspace(en.min()-1,en.max()+1,yd)
if 1:
    for ix,x in enumerate(KKs):
        for iy,y in enumerate(YY):
            for i in range(pts):
                for j in range(4):
                    lor[ix,iy] += abs(weight[i,j])/((x-ks[i])**2+K_**2)/((y-en[i,j])**2+E_**2)
lor_ = lor[:-1,:-1]
X,Y = np.meshgrid(KKs,YY)
plt.pcolormesh(X, Y,lor_.T,norm=LogNorm(vmin=lor.min(), vmax=lor.max()), cmap=plt.cm.plasma)
#plt.scatter(X,Y,c=lor.T)
plt.colorbar()
plt.show()
