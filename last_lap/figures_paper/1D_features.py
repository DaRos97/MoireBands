"""
Here we plot the 1D features of side bands

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import sys

mass = 1

def Hamiltonian(k,V,phi,N,G):
    """
    Simply the Hamiltonian, which has the dispersion in the diagonal entries (quadratic here for simplicity) and in the first diagonal the moire potential coupling.
    """
    H = np.zeros((2*N+1,2*N+1),dtype=complex)
    for n in range(-N,N):
        H[n+N,n+N] = -(k+n*G)**2/2/mass
        H[n+N,n+N+1] = V*np.exp(1j*phi)
        H[n+N+1,n+N] = V*np.exp(-1j*phi)
    H[2*N,2*N] = -(k+N*G)**2/2/mass
    return H
#Gap
def ind_k(k_pt,list_momenta):
    """
    Index of k_pt in momentum list.
    """
    initial_momentum = list_momenta[0]
    final_momentum = list_momenta[-1]
    momentum_points = len(list_momenta)
    return int(momentum_points*(k_pt-initial_momentum)/(final_momentum-initial_momentum))
def gap(energies,G_pt,level,list_momenta):
    """
    Energy gap between two neighboring bands. 
    level specifies which bands from the top are considered.
    G_pt specifies in which momentum the gap is considered.
    """
    return (energies[-1-level]-energies[-2-level])[ind_k(G_pt,list_momenta)]
#band distance
def horizontal_displacement(e_,energies,weights,list_momenta):
    """
    Compute indeces of main and first two side bands, using the weights to discriminate.
    """
    delta_e = 1/2/mass*np.sqrt(2*mass*abs(e_))*(list_momenta[1]-list_momenta[0])
    l = np.argwhere(abs(e_-energies[:,:momentum_points//2])<delta_e)
    indices = np.argsort([weights[l[i,0],l[i,1]] for i in range(l.shape[0])])
    filtered_indices = [indices[-1],]
    for i in range(indices.shape[0]):
        temp = indices[-1-i]
        if abs(l[temp,1]-l[filtered_indices[-1],1])>2:
                filtered_indices.append(temp)
    i_mb = filtered_indices[0]
    i_sb1 = filtered_indices[1]
    i_sb2 = filtered_indices[2]
    return l,(i_mb,i_sb1,i_sb2)

def vertical_displacement(i_k,weights):
    indices = np.argsort(weights[:,i_k])
    i_mb = indices[-1]
    i_sb2 = indices[-2]
    i_sb1 = indices[-3] if indices[-3] < indices[-2] else indices[-4]
    return (i_mb,i_sb1,i_sb2)

def plot_single_parameter_set(e_,energies,weights,mrl,list_momenta,title):
    """
    Given energy e_ and a set of eigenvalues and weights computes the image.
    """
    #Figure
    fig,ax = plt.subplots()
    fig.set_size_inches(18,12)
    for t in range(2*side_bands+1):
        ax.plot(list_momenta,energies[t],color='k',lw=LW)
        ax.scatter(list_momenta,energies[t],s=weights[t]*70,c='b',lw=0)
    #Gap arrows
    ax.arrow(-mrl/2,energies[-1][ind_k(-mrl/2,list_momenta)],
            0,-gap(energies,-mrl/2,0,list_momenta),
            color='r',label='first gap',head_length=0,width=0)
#    ax.arrow(-mrl,energies[-2][ind_k(-mrl,list_momenta)],
#            0,-gap(energies,-mrl,1,list_momenta),
#            color='y',label='gap 2',head_length=0,width=0)
    #Horizontal distance arrow
    l, inds = horizontal_displacement(e_,energies,weights,list_momenta)
    i_mb,i_sb1,i_sb2 = inds
    ax.scatter(list_momenta[l[i_mb,1]],energies[l[i_mb,0],l[i_mb,1]],c='k',s=150)
    ax.scatter(list_momenta[l[i_sb1,1]],energies[l[i_sb1,0],l[i_sb1,1]],c='lime',s=100)
    ax.scatter(list_momenta[l[i_sb2,1]],energies[l[i_sb2,0],l[i_sb2,1]],c='g',s=100)

    ax.arrow(list_momenta[l[i_mb,1]],energies[l[i_mb,0],l[i_mb,1]],
            list_momenta[l[i_sb1,1]]-list_momenta[l[i_mb,1]],0,
            color='lime',label='inner band',head_length=0,width=0)
    ax.arrow(list_momenta[l[i_mb,1]],energies[l[i_mb,0],l[i_mb,1]],
            list_momenta[l[i_sb2,1]]-list_momenta[l[i_mb,1]],0,
            color='g',label='outer band',head_length=0,width=0)
    #Vertical distance arrow
    i_k = l[i_mb,1]
    i_mb,i_sb1,i_sb2 = vertical_displacement(i_k,weights)
    ax.scatter(list_momenta[i_k],energies[i_mb,i_k],c='m',s=150,zorder=10)
    ax.scatter(list_momenta[i_k],energies[i_sb1,i_k],c='aqua',s=100)
    ax.scatter(list_momenta[i_k],energies[i_sb2,i_k],c='dodgerblue',s=100)

    ax.arrow(list_momenta[i_k],energies[i_mb,i_k],     #x,y,dx,dy
            0,energies[i_sb1,i_k]-energies[i_mb,i_k],
            color='aqua',label='lower band',head_length=0,width=0)
    ax.arrow(list_momenta[i_k],energies[i_mb,i_k],
            0,energies[i_sb2,i_k]-energies[i_mb,i_k],
            color='dodgerblue',label='upper band',head_length=0,width=0)

    #Plot features
    ax.legend(fontsize=20,loc='upper right')
    #ax.set_title(title)
    #Limits
    rg = np.max(energies[-1])-np.min(energies[side_bands])
    ax.set_ylim(e_*3,np.max(energies[-1])*2+0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

#Parameters

side_bands = 4
momentum_points = 1001
c = ['k','g','r','r','y','m','c']
LW = 0.1    #line width

#######################################################################################################
#######################################################################################################
#######################################################################################################
#Many V and aM
V_list = np.linspace(0.015,0.02,1)
aM_list = np.linspace(17,25,1)
phi = 0

gaps1 = np.zeros((len(V_list),len(aM_list)))
gaps2 = np.zeros((len(V_list),len(aM_list)))
h_disp1 = np.zeros((len(V_list),len(aM_list),2))    #the last dimension (2) is for displacement and relative weight
h_disp2 = np.zeros((len(V_list),len(aM_list),2))
v_disp1 = np.zeros((len(V_list),len(aM_list),2))
v_disp2 = np.zeros((len(V_list),len(aM_list),2))

N = side_bands
for v in range(len(V_list)):
    for aM in range(len(aM_list)):
        G = np.pi*2/aM_list[aM]
        k_i = -4*G
        k_f = 4*G
        list_momenta = np.linspace(k_i,k_f,momentum_points)
        en = np.zeros((2*N+1,momentum_points))
        wp = np.zeros((2*N+1,momentum_points))
        for ii,k in enumerate(list_momenta):
            en[:,ii],ev = np.linalg.eigh(Hamiltonian(k,V_list[v],phi,N,G))
            wp[:,ii] = np.absolute(ev[N])**2
        gaps1[v,aM] = gap(en,-G/2,0,list_momenta)
        gaps2[v,aM] = gap(en,-G,1,list_momenta)
        #
        e_ = -(-1.7*G)**2/2/mass        #energy of main band between 2nd and third side band max
        l, inds = horizontal_displacement(e_,en,wp,list_momenta)
        i_mb,i_sb1,i_sb2 = inds
        h_disp1[v,aM,0] = abs(list_momenta[l[i_sb1,1]]-list_momenta[l[i_mb,1]])
        h_disp1[v,aM,1] = wp[l[i_sb1,0],l[i_sb1,1]]/wp[l[i_mb,0],l[i_mb,1]]
        h_disp2[v,aM,0] = abs(list_momenta[l[i_sb2,1]]-list_momenta[l[i_mb,1]])
        h_disp2[v,aM,1] = wp[l[i_sb2,0],l[i_sb2,1]]/wp[l[i_mb,0],l[i_mb,1]]
        #
        i_k = l[i_mb,1]
        i_mb,i_sb1,i_sb2 = vertical_displacement(i_k,wp)
        v_disp1[v,aM,0] = abs(en[i_sb1,i_k]-en[i_mb,i_k])
        v_disp1[v,aM,1] = wp[i_sb1,i_k]/wp[i_mb,i_k]
        v_disp2[v,aM,0] = abs(en[i_sb2,i_k]-en[i_mb,i_k])
        v_disp2[v,aM,1] = wp[i_sb2,i_k]/wp[i_mb,i_k]
        if 1:
            plot_single_parameter_set(e_,en,wp,G,list_momenta,"aM="+"{:.1f}".format(aM_list[aM])+", V="+"{:.3f}".format(V_list[v]))

s_ = 20
if 1:
    fig,axs = plt.subplots(1,3)
    fig.set_size_inches(25,8)
    cmap = mpl.colormaps['viridis']
    #One
    ax = axs[0]
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,gaps1[v,:],label="V="+"{:.3f}".format(V_list[v]),color=colors[v])
    ax.set_xlabel("aM",fontsize=s_)
    ax.set_ylabel("gap -",fontsize=s_)
    ax.legend(fontsize=s_)

    #Two
    ax = axs[1]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,h_disp1[v,:,0],ls='dotted',color=colors[v],label='horizontal inner')
        ax.plot(aM_list,h_disp2[v,:,0],ls=(0,(1,10)),color=colors[v],label='horizontal outer')
        ax.plot(aM_list,v_disp1[v,:,0],ls='dashed',color=colors[v],label='vertical inner')
        ax.plot(aM_list,v_disp2[v,:,0],ls='dashdot',color=colors[v],label='vertical outer')
    ax.set_xlabel("aM",fontsize=s_)
    han,lab = ax.get_legend_handles_labels()
    ax.legend(handles=han[:4],labels=lab[:4],fontsize=s_)
    ax.set_ylabel("displacement",fontsize=s_)

    #Three
    ax = axs[2]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,h_disp1[v,:,1],ls='dotted',color=colors[v])
        ax.plot(aM_list,h_disp2[v,:,1],ls=(0,(1,10)),color=colors[v])
        ax.plot(aM_list,v_disp1[v,:,1],ls='dashed',color=colors[v])
        ax.plot(aM_list,v_disp2[v,:,1],ls='dashdot',color=colors[v])
    ax.set_xlabel("aM",fontsize=s_)
    ax.legend(handles=han[:4],labels=lab[:4],fontsize=s_)
    ax.set_ylabel("Relative weight",fontsize=s_)

    fig.tight_layout()
    plt.show()
else:
    fig,axs = plt.subplots(1,3)
    fig.set_size_inches(25,12)
    cmap = mpl.colormaps['viridis']
    #One
    ax = axs[0]
    colors = cmap(np.linspace(0,1,len(aM_list)))
    for aM in range(len(aM_list)):
        ax.plot(V_list,gaps1[:,aM],label="aM="+"{:.3f}".format(aM_list[aM]),color=colors[aM])
    ax.set_xlabel("V")
    ax.set_ylabel("gap -")
    ax.legend()

    #Two
    ax = axs[1]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(aM_list)))
    for aM in range(0,len(aM_list),2):
        ax.plot(V_list,h_disp1[:,aM,0],ls='dotted',color=colors[aM])
        ax.plot(V_list,h_disp2[:,aM,0],ls=(0,(1,10)),color=colors[aM])
        ax_r.plot(V_list,v_disp1[:,aM,0],ls='dashed',color=colors[aM])
        ax_r.plot(V_list,v_disp2[:,aM,0],ls='dashdot',color=colors[aM])
    ax.set_xlabel("V")
    ax.set_ylabel("horizontal displacement inner (..) and outer (. . .) bands -")
    ax_r.set_ylabel("vertical displacement inner (_ _) and outer (_ . _) bands -")

    #Three
    ax = axs[2]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(aM_list)))
    for aM in range(0,len(aM_list),2):
        ax.plot(V_list,h_disp1[:,aM,1],ls='dotted',color=colors[aM])
        ax.plot(V_list,h_disp2[:,aM,1],ls=(0,(1,10)),color=colors[aM])
        ax_r.plot(V_list,v_disp1[:,aM,1],ls='dashed',color=colors[aM])
        ax_r.plot(V_list,v_disp2[:,aM,1],ls='dashdot',color=colors[aM])
    ax.set_xlabel("V")
    ax.set_ylabel("horizontal RW inner (..) and outer (. . .) bands -")
    ax_r.set_ylabel("vertical RW inner (_ _) and outer (_ . _) bands -")

    fig.tight_layout()
    plt.show()

