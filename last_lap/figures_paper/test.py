"""
Here we plot the 1D features

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

def H(k,V,phi,N,G):
    HH = np.zeros((2*N+1,2*N+1),dtype=complex)
    for n in range(-N,N):
        HH[n+N,n+N] = -(k+n*G)**2
        HH[n+N,n+N+1] = V*np.exp(1j*phi)
        HH[n+N+1,n+N] = V*np.exp(-1j*phi)
    HH[2*N,2*N] = -(k+N*G)**2
    return HH
#Gap
def ind_k(k_pt):
    """
    Index of k_pt in momentum list.
    """
    return int(momentum_points*(k_pt-initial_momentum)/(final_momentum-initial_momentum))
def gap(energies,G_pt,level):
    """
    Energy gap between two neighboring bands. 
    level specifies which bands from the top are considered.
    G_pt specifies in which momentum the gap is considered.
    """
    return (energies[-1-level]-energies[-2-level])[ind_k(G_pt)]
#band distance
def horizontal_band_distance(energies,e_):
    """
    Difference in momentum (distance) between main band and proximate external band.
    e_ is energy at which the distance is computed, to be defined depending on the other parameters.
    """
    ik_main = np.argmin(abs(e_-energies[side_bands]))
    ik_side = np.argmin(abs(e_-energies[side_bands+1]))
    return list_momenta[ik_main]-list_momenta[ik_side]
def horizontal_RW(energies,weights,e_):
    """
    Relative weigth of side band wrt main one, at fixed energy.
    e_ is energy at which the RW is computed, to be defined depending on the other parameters.
    This works only if the energy considered is low enough so that the main band is exactly at index N.
    """
    ik_main = np.argmin(abs(e_-energies[N]))
    ik_side = np.argmin(abs(e_-energies[N+1]))
    return weights[N+1,ik_main]/weights[N,ik_side]

side_bands = 5

momentum_points = 501
c = ['k','g','r','r','y','m','c']
LW = 0.1    #line width

#######################################################################################################
#######################################################################################################
#######################################################################################################
#Energy and weights of one set of parameters V,phi,a_M
#Input values
moire_potential_amplitude = float(sys.argv[1]) if len(sys.argv)>1 else 0.1 
moire_potential_phase = float(sys.argv[2]) if len(sys.argv)>2 else 0
moire_lattice_length = float(sys.argv[3]) if len(sys.argv)>3 else 10
#Derived values
moire_reciprocal_lattice = np.pi*2/moire_lattice_length
initial_momentum = -(side_bands+1)*moire_reciprocal_lattice
final_momentum = (side_bands+1)*moire_reciprocal_lattice
list_momenta = np.linspace(initial_momentum,final_momentum,momentum_points)
#Enegries and weights
energies = np.zeros((2*side_bands+1,momentum_points))
weights = np.zeros((2*side_bands+1,momentum_points))
for ii,k in enumerate(list_momenta):
    energies[:,ii],ev = np.linalg.eigh(H(k,moire_potential_amplitude,moire_potential_phase,side_bands,moire_reciprocal_lattice))
    weights[:,ii] = np.absolute(ev[side_bands])**2

#Figure
fig,axs = plt.subplots(2,1)
fig.set_size_inches(18,12)
#Fig 1
ax = axs[0]
for t in range(2*side_bands+1):
    ax.plot(list_momenta,energies[t],color='k',lw=LW)
    ax.scatter(list_momenta,energies[t],s=weights[t]*20,c='b',lw=0)
#Gap arrows
ax.arrow(-moire_reciprocal_lattice/2,energies[-1][ind_k(-moire_reciprocal_lattice/2)],
        0,-gap(energies,-moire_reciprocal_lattice/2,0),
        color='r',label='gap 1',head_length=0,width=0)
ax.arrow(-moire_reciprocal_lattice,energies[-2][ind_k(-moire_reciprocal_lattice)],
        0,-gap(energies,-moire_reciprocal_lattice,1),
        color='y',label='gap 2',head_length=0,width=0)
#Distance arrow
e_ = (np.max(energies[side_bands,ind_k(-5/2*moire_reciprocal_lattice):ind_k(-3/2*moire_reciprocal_lattice)])+np.min(energies[side_bands,ind_k(-2*moire_reciprocal_lattice):ind_k(-moire_reciprocal_lattice)]))/2
r_en = energies[:,:ind_k(-2*moire_reciprocal_lattice)]
ind_e = np.argmin(abs(e_-r_en[side_bands]))
ax.arrow(list_momenta[ind_e],r_en[side_bands][ind_e],
        -horizontal_band_distance(r_en,e_),0,
        color='g',label='displacement',head_length=0,width=0)
#Plot features
ax.legend()
ax.set_title("aM="+"{:.1f}".format(moire_lattice_length)+", V="+"{:.3f}".format(moire_potential_amplitude))
#Limits
rg = np.max(energies[-1])-np.min(energies[side_bands])
ax.set_ylim(r_en[side_bands][ind_e]*1.5,np.max(energies[-1])+rg/10)
ax.set_xlim(-4*moire_reciprocal_lattice,4*moire_reciprocal_lattice)
plt.show()
exit()

#######################################################################################################
#######################################################################################################
#######################################################################################################
#Many V and aM
V_list = np.linspace(0,0.01,5)
aM_list = np.linspace(10,100,5)

gaps1 = np.zeros((len(V_list),len(aM_list)))
gaps2 = np.zeros((len(V_list),len(aM_list)))
disp = np.zeros((len(V_list),len(aM_list)))
inte = np.zeros((len(V_list),len(aM_list)))
for v in range(len(V_list)):
    for aM in range(len(aM_list)):
        G = np.pi*2/aM_list[aM]
        ki = -(N+1)*G
        kf = (N+1)*G
        k_list = np.linspace(ki,kf,kpts)
        en = np.zeros((2*N+1,kpts))
        wp = np.zeros((2*N+1,kpts))
        for ii,k in enumerate(k_list):
            en[:,ii],ev = np.linalg.eigh(H(k,V_list[v],N,G))
            wp[:,ii] = np.absolute(ev[N])**2
        gaps1[v,aM] = gap(en,-G/2,0)
        gaps2[v,aM] = gap(en,-G,1)
        if en[N,ind_k(-2*G)]>en[N,ind_k(-3/2*G)]:
            e_ = (np.max(en[N,ind_k(-5/2*G):ind_k(-3/2*G)])+np.min(en[N,ind_k(-2*G):ind_k(-G)]))/2
            disp[v,aM] = band_distance(en[:,:ind_k(-2*G)],e_)
            inte[v,aM] = rel_weight(en[:,:ind_k(-2*G)],wp,e_)
        else:
            disp[v,aM] = np.nan
            inte[v,aM] = np.nan

#Fig 2
cmap = mpl.colormaps['plasma']
colors = cmap(np.linspace(0,1,len(aM_list)))
ax = axs[1]
ax_r = ax.twinx()
for aM in range(len(aM_list)):
    ax.plot(V_list,gaps1[:,aM],label="aM="+"{:.1f}".format(aM_list[aM]),color=colors[aM])
#    ax.plot(V_list,gaps2[:,aM],ls='--',color=colors[aM])
    ax_r.plot(V_list,disp[:,aM],ls='-.',color=colors[aM])
    ax_r.plot(V_list,inte[:,aM],ls=':',color=colors[aM])
ax.set_xlabel("V")
ax.set_ylabel("gap -")
ax_r.set_yticks([])
ax.legend()

#Fig 3
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0,1,len(V_list)))
ax = axs[2]
ax_r = ax.twinx()
for v in range(len(V_list)):
    ax.plot(aM_list,gaps1[v],label="V="+"{:.3f}".format(V_list[v]),color=colors[v])
#    ax.plot(aM_list,gaps2[v],ls='--',color=colors[v])
    ax_r.plot(aM_list,disp[v],ls='-.',color=colors[v])
    ax_r.plot(aM_list,inte[v],ls=':',color=colors[v])
ax.set_xlabel("aM")
ax.set_yticks([])
ax_r.set_ylabel("displacement -.- and relative weight ..")
ax.legend()

plt.show()


