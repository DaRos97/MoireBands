import numpy as np
import sys
import getopt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm


fig = plt.figure(figsize = (15,8))
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
if plot_mono:
    for b in range(14):#len(res_mono_LL[0])):      #plot valence bands (2 for spin-rbit) of monolayer
        plt.plot(K_list,res_mono_LL[:,b],'g-',lw = 0.5)
    for b in range(14):#len(res_mono_UL[0])):      #plot valence bands (2 for spin-rbit) of monolayer
        plt.plot(K_list,res_mono_UL[:,b]-offset_energy,'r-',lw = 0.5)
for i,c in enumerate([*Path]):      #plot symmetry points as vertical lines
    a = 1 if i == len(Path)-1 else 0
    plt.vlines(K_list[i*lp//(len(Path)-1)-a],MIN_E,MAX_E,'k',lw=0.3,label=c)
    plt.text(K_list[i*lp//(len(Path)-1)-a],MIN_E-delta/12,dic_sym[c])
#
X,Y = np.meshgrid(K_list,E_list)
#for i in range(bnds):
#    plt.scatter(K_list,res[0,:,i],s=1)
plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=lor[np.nonzero(lor)].min(), vmax=lor.max()))
plt.ylabel('eV')
plt.ylim(-2,0.5)
plt.show()
exit()
ax = fig.add_subplot(122)
ax.axes.get_xaxis().set_visible(False)
plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=lor[np.nonzero(lor)].min(), vmax=lor.max()))
plt.ylim(-2,0.5)
plt.show()
