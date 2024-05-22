import numpy as np
import matplotlib.pyplot as plt
import functions as fs

a_M = 1/np.sqrt(1/fs.a_WSe2**2+1/fs.a_WS2**2-2/fs.a_WSe2/fs.a_WS2)
#Moire parameters
N = 2                                #####################
n_cells = int(1+3*N*(N+1))
#Model parameters
m1,m2,mu = (0.13533,0.53226,-1.16385)

a,b,c = (0,0,2)
type_of_stacking = 'P'
V,phi = (0.03,np.pi)

all_pars = (type_of_stacking,m1,m2,mu,a,b,c,N,V,phi)
G_M = fs.get_Moire(a_M)

title = ''
title += 'm1: '+"{:.3f}".format(m1)+', m2: '+"{:.3f}".format(m2)+r', $\mu$: '+"{:.3f}".format(mu)+'\n'
title += 'a: '+"{:.3f}".format(a)+', b: '+"{:.3f}".format(b)+', c: '+"{:.3f}".format(c)+'\n'
title += 'V: '+"{:.3f}".format(V)+r', $\phi$: '+"{:.3f}".format(phi)+'\n'

k_pts = 100
k_list = np.zeros((k_pts,2))
k_list[:,0] = np.linspace(-1,1,k_pts)
print(title)

energies = np.zeros((k_pts,2*n_cells))
weights = np.zeros((k_pts,2*n_cells))
lu = fs.lu_table(N)
for i in range(k_pts):
    H = fs.big_H(k_list[i],lu,all_pars,G_M)
    energies[i,:],ev = np.linalg.eigh(H)
    ab = np.absolute(ev)**2
    weights[i,:] = np.sum(ab[0,:],axis=0) + np.sum(ab[n_cells,:],axis=0)


plt.figure(figsize=(20,15))
for e in range(2*n_cells):
    e_line = energies[:,e]
    plt.plot(k_list[:,0],e_line,color='r',zorder=1,linewidth=0.1)
    plt.scatter(k_list[:,0],e_line,s=weights[:,e],lw=0,color='r',marker='o',zorder=3)

plt.show()
