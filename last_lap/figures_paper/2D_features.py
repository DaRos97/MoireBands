import numpy as np
import functions as fs
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm


cut = 'KGK' #'GMG'
k_pts = 301
K_list = fs.get_K(cut,k_pts)
abs_k = np.array([np.linalg.norm(K_list[i])*np.sign(K_list[i,0]) for i in range(K_list.shape[0])])
mass = 0.5
N = 3
n_cells = int(1+3*N*(N+1))

V_list = np.linspace(0.001,0.02,3)
aM_list = np.linspace(4,25,3)
phi_list = np.linspace(0,2*np.pi,10)

gaps1 = np.zeros((len(V_list),len(aM_list),len(phi_list)))
gaps2 = np.zeros((len(V_list),len(aM_list),len(phi_list)))
h_disp1 = np.zeros((len(V_list),len(aM_list),len(phi_list),2))
h_disp2 = np.zeros((len(V_list),len(aM_list),len(phi_list),2))
v_disp1 = np.zeros((len(V_list),len(aM_list),len(phi_list),2))
v_disp2 = np.zeros((len(V_list),len(aM_list),len(phi_list),2))

for v in tqdm(range(len(V_list))):
    for a in tqdm(range(len(aM_list))):
        for p in range(len(phi_list)):
            # One parameter set evaluation
            V = V_list[v]     #1
            a_Moire = aM_list[a]    #2
            phi = phi_list[p]         #3

            G_M = [0,4*np.pi/np.sqrt(3)/a_Moire*np.array([0,1])]    
            G_M[0] = np.tensordot(fs.R_z(-np.pi/3),G_M[1],1)
            G = np.linalg.norm(G_M[0])
            #Compute energies and weights along KGK
            energies = np.zeros((k_pts,n_cells))
            weights = np.zeros((k_pts,n_cells))
            look_up = fs.lu_table(N)
            for i in range(k_pts):
                H_tot = fs.big_H(K_list[i],look_up,(N,V,phi,mass),G_M)
                energies[i,:],evecs = np.linalg.eigh(H_tot)           #Diagonalize to get eigenvalues and eigenvectors
                for e in range(n_cells):
                    weights[i,e] = np.abs(evecs[0,e])**2
            #
            gap_k = -G/np.sqrt(3) if cut == 'KGK' else -G/2
            i_k = fs.ind_k(gap_k,K_list)
            gaps1[v,a,p] = fs.gap(energies,weights,i_k,K_list)[0]
            #
            e_ = -(-1.7*G)**2/2/mass        #energy of main band between 2nd and third side band max
            l, inds = fs.horizontal_displacement(e_,energies,weights,K_list,mass)
            i_mb,i_sb1,i_sb2 = inds
            h_disp1[v,a,p,0] = abs(abs_k[l[i_sb1,0]]-abs_k[l[i_mb,0]])
            h_disp1[v,a,p,1] = weights[l[i_sb1,0],l[i_sb1,1]]/weights[l[i_mb,0],l[i_mb,1]]
            h_disp2[v,a,p,0] = abs(abs_k[l[i_sb2,0]]-abs_k[l[i_mb,0]])
            h_disp2[v,a,p,1] = weights[l[i_sb2,0],l[i_sb2,1]]/weights[l[i_mb,0],l[i_mb,1]]
            #
            i_k = l[i_mb,0]
            i_mb,i_sb1,i_sb2 = fs.vertical_displacement(i_k,weights)
            v_disp1[v,a,p,0] = abs(energies[i_k,i_sb1]-energies[i_k,i_mb])
            v_disp1[v,a,p,1] = weights[i_k,i_sb1]/weights[i_k,i_mb]
            v_disp2[v,a,p,0] = abs(energies[i_k,i_sb2]-energies[i_k,i_mb])
            v_disp2[v,a,p,1] = weights[i_k,i_sb2]/weights[i_k,i_mb]
            
            if 0:
                pars = (N,V,phi,mass,G,cut)
                fs.plot_single_parameter_set(-5,energies,weights,pars,K_list,'aaa')

#Plot
s_ = 20
if 1:
    fig,axs = plt.subplots(1,3)
    fig.set_size_inches(25,8)
    cmap = mpl.colormaps['viridis']
    #One
    ax = axs[0]
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,gaps1[v,:,0],label="V="+"{:.3f}".format(V_list[v]),color=colors[v])
    ax.set_xlabel("aM",fontsize=s_)
    ax.set_ylabel("gap -",fontsize=s_)
    ax.legend(fontsize=s_)

    #Two
    ax = axs[1]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,h_disp1[v,:,0,0],ls='dotted',color=colors[v],label='horizontal inner')
        ax.plot(aM_list,h_disp2[v,:,0,0],ls=(0,(1,10)),color=colors[v],label='horizontal outer')
        ax.plot(aM_list,v_disp1[v,:,0,0],ls='dashed',color=colors[v],label='vertical inner')
        ax.plot(aM_list,v_disp2[v,:,0,0],ls='dashdot',color=colors[v],label='vertical outer')
    ax.set_xlabel("aM",fontsize=s_)
    han,lab = ax.get_legend_handles_labels()
    ax.legend(handles=han[:4],labels=lab[:4],fontsize=s_)
    ax.set_ylabel("displacement",fontsize=s_)

    #Three
    ax = axs[2]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,h_disp1[v,:,0,1],ls='dotted',color=colors[v])
        ax.plot(aM_list,h_disp2[v,:,0,1],ls=(0,(1,10)),color=colors[v])
        ax.plot(aM_list,v_disp1[v,:,0,1],ls='dashed',color=colors[v])
        ax.plot(aM_list,v_disp2[v,:,0,1],ls='dashdot',color=colors[v])
    ax.set_xlabel("aM",fontsize=s_)
    ax.legend(handles=han[:4],labels=lab[:4],fontsize=s_)
    ax.set_ylabel("Relative weight",fontsize=s_)

    fig.tight_layout()
    plt.show()





