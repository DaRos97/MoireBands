import numpy as np
import functions as fs
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from pathlib import Path

cut = 'MGM'
k_pts = 1001
mass = 1
N = 3
n_cells = int(1+3*N*(N+1))
save = False

V_list = np.linspace(0.05,0.05,2)
aM_list = np.linspace(4,6,1)
phi_list = np.linspace(0,np.pi,101)

fn_g1 = 'data_'+cut+'/gap1_V('+"{:.3f}".format(V_list[0])+'_'+"{:.3f}".format(V_list[-1])+'_'+str(len(V_list))+')_aM('+str(aM_list[0])+'_'+str(aM_list[-1])+'_'+str(len(aM_list))+')_phi('+"{:.3f}".format(phi_list[0])+'_'+"{:.3f}".format(phi_list[-1])+'_'+str(len(phi_list))+').npy'
fn_g2 = 'data_'+cut+'/gap2_V('+"{:.3f}".format(V_list[0])+'_'+"{:.3f}".format(V_list[-1])+'_'+str(len(V_list))+')_aM('+str(aM_list[0])+'_'+str(aM_list[-1])+'_'+str(len(aM_list))+')_phi('+"{:.3f}".format(phi_list[0])+'_'+"{:.3f}".format(phi_list[-1])+'_'+str(len(phi_list))+').npy'
fn_h1 = 'data_'+cut+'/h1_V('+"{:.3f}".format(V_list[0])+'_'+"{:.3f}".format(V_list[-1])+'_'+str(len(V_list))+')_aM('+str(aM_list[0])+'_'+str(aM_list[-1])+'_'+str(len(aM_list))+')_phi('+"{:.3f}".format(phi_list[0])+'_'+"{:.3f}".format(phi_list[-1])+'_'+str(len(phi_list))+').npy'
fn_h2 = 'data_'+cut+'/h2_V('+"{:.3f}".format(V_list[0])+'_'+"{:.3f}".format(V_list[-1])+'_'+str(len(V_list))+')_aM('+str(aM_list[0])+'_'+str(aM_list[-1])+'_'+str(len(aM_list))+')_phi('+"{:.3f}".format(phi_list[0])+'_'+"{:.3f}".format(phi_list[-1])+'_'+str(len(phi_list))+').npy'
fn_v1 = 'data_'+cut+'/v1_V('+"{:.3f}".format(V_list[0])+'_'+"{:.3f}".format(V_list[-1])+'_'+str(len(V_list))+')_aM('+str(aM_list[0])+'_'+str(aM_list[-1])+'_'+str(len(aM_list))+')_phi('+"{:.3f}".format(phi_list[0])+'_'+"{:.3f}".format(phi_list[-1])+'_'+str(len(phi_list))+').npy'
fn_v2 = 'data_'+cut+'/v2_V('+"{:.3f}".format(V_list[0])+'_'+"{:.3f}".format(V_list[-1])+'_'+str(len(V_list))+')_aM('+str(aM_list[0])+'_'+str(aM_list[-1])+'_'+str(len(aM_list))+')_phi('+"{:.3f}".format(phi_list[0])+'_'+"{:.3f}".format(phi_list[-1])+'_'+str(len(phi_list))+').npy'

fns = [fn_g1,fn_g2,fn_h1,fn_h2,fn_v1,fn_v2]
compute = False
for fn in fns:
    if not Path(fn).is_file():
        compute = True

if compute:
    gaps1 = np.zeros((len(V_list),len(aM_list),len(phi_list)))
    gaps2 = np.zeros((len(V_list),len(aM_list),len(phi_list)))
    h_disp1 = np.zeros((len(V_list),len(aM_list),len(phi_list),2))
    h_disp2 = np.zeros((len(V_list),len(aM_list),len(phi_list),2))
    v_disp1 = np.zeros((len(V_list),len(aM_list),len(phi_list),2))
    v_disp2 = np.zeros((len(V_list),len(aM_list),len(phi_list),2))

    for v in range(len(V_list)):
        print("V: ",v,V_list[v])
        for a in range(len(aM_list)):
            print("aM: ",a,aM_list[a])
            for p in range(len(phi_list)):
                print("phi: ",p,phi_list[p])
                # One parameter set evaluation
                V = V_list[v]     #1
                a_Moire = aM_list[a]    #2
                phi = phi_list[p]         #3
                #
                K_list = fs.get_K(cut,k_pts,a_Moire)
                abs_k = np.array([np.linalg.norm(K_list[i])*np.sign(K_list[i,0]) for i in range(K_list.shape[0])])
                #
                G_M = np.array([np.array([1,1/np.sqrt(3)]),np.array([0,2/np.sqrt(3)])])*2*np.pi/a_Moire
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
                k_pt = np.array([-G/np.sqrt(3),0]) if cut == 'KGK' else -G_M[0]/2
                i_k = fs.ind_k(k_pt,K_list)
                gap_k = np.linalg.norm(k_pt)
                gaps1[v,a,p] = fs.gap(energies,weights,i_k,cut)[0]
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
                
                if not save:
                    pars = (N,V,phi,mass,G,cut,G_M)
                    fs.plot_single_parameter_set(e_,energies,weights,pars,K_list,'Cut: '+cut)
    if save:
        np.save(fn_g1,gaps1)
        np.save(fn_g2,gaps2)
        np.save(fn_h1,h_disp1)
        np.save(fn_h2,h_disp2)
        np.save(fn_v1,v_disp1)
        np.save(fn_v2,v_disp2)
else:
    gaps1 = np.load(fn_g1)
    gaps2 = np.load(fn_g2)
    h_disp1 = np.load(fn_h1)
    h_disp2 = np.load(fn_h2)
    v_disp1 = np.load(fn_v1)
    v_disp2 = np.load(fn_v2)

#Plot
s_ = 20
if 0:   #"fixed phi, function of aM"
    fig,axs = plt.subplots(1,3)
    fig.set_size_inches(25,8)
    cmap = mpl.colormaps['cool']

    i_phi = 0
    #One
    ax = axs[0]
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,gaps1[v,:,i_phi],label="V="+"{:.3f}".format(V_list[v]),color=colors[v])
    ax.set_xlabel("aM",fontsize=s_)
    ax.set_ylabel("gap -",fontsize=s_)
    ax.legend(fontsize=s_)

    #Two
    ax = axs[1]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,h_disp1[v,:,i_phi,0],ls='dotted',color=colors[v],label='external band')
        ax.plot(aM_list,h_disp2[v,:,i_phi,0],ls=(0,(1,10)),color=colors[v],label='inner band')
        ax.plot(aM_list,v_disp1[v,:,i_phi,0],ls='dashed',color=colors[v],label='lower band')
        ax.plot(aM_list,v_disp2[v,:,i_phi,0],ls='dashdot',color=colors[v],label='higher band')
    ax.set_xlabel("aM",fontsize=s_)
    han,lab = ax.get_legend_handles_labels()
    ax.legend(handles=han[:4],labels=lab[:4],fontsize=s_)
    ax.set_ylabel("displacement",fontsize=s_)

    #Three
    ax = axs[2]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(V_list)))
    for v in range(0,len(V_list),2):
        ax.plot(aM_list,h_disp1[v,:,i_phi,1],ls='dotted',color=colors[v])
        ax.plot(aM_list,h_disp2[v,:,i_phi,1],ls=(0,(1,10)),color=colors[v])
        ax.plot(aM_list,v_disp1[v,:,i_phi,1],ls='dashed',color=colors[v])
        ax.plot(aM_list,v_disp2[v,:,i_phi,1],ls='dashdot',color=colors[v])
    ax.set_xlabel("aM",fontsize=s_)
    ax.legend(handles=han[:4],labels=lab[:4],fontsize=s_)
    ax.set_ylabel("Relative weight",fontsize=s_)
    
    fig.tight_layout()
    plt.suptitle("cut "+cut+", phi="+"{:.3f}".format(phi_list[i_phi]))
    plt.show()
if 0:   #"fixed phi, function of V"
    fig,axs = plt.subplots(1,3)
    fig.set_size_inches(25,8)
    cmap = mpl.colormaps['cool']

    i_phi = 0
    #One
    ax = axs[0]
    colors = cmap(np.linspace(0,1,len(aM_list)))
    for a in range(0,len(aM_list)):
        ax.plot(V_list,gaps1[:,a,i_phi],label="aM="+str(aM_list[a]),color=colors[a])
    ax.set_xlabel("V",fontsize=s_)
    ax.set_ylabel("gap -",fontsize=s_)
    ax.legend(fontsize=s_)

    #Two
    ax = axs[1]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(aM_list)))
    for a in range(0,len(aM_list)):
        ax.plot(V_list,h_disp1[:,a,i_phi,0],ls='dotted',color=colors[a],label='external band')
        ax.plot(V_list,h_disp2[:,a,i_phi,0],ls=(0,(1,10)),color=colors[a],label='inner band')
        ax.plot(V_list,v_disp1[:,a,i_phi,0],ls='dashed',color=colors[a],label='lower band')
        ax.plot(V_list,v_disp2[:,a,i_phi,0],ls='dashdot',color=colors[a],label='higher band')
    ax.set_xlabel("V",fontsize=s_)
    han,lab = ax.get_legend_handles_labels()
    ax.legend(handles=han[:4],labels=lab[:4],fontsize=s_)
    ax.set_ylabel("displacement",fontsize=s_)

    #Three
    ax = axs[2]
    ax_r = ax.twinx()
    colors = cmap(np.linspace(0,1,len(aM_list)))
    for a in range(0,len(aM_list)):
        ax.plot(V_list,h_disp1[:,a,i_phi,1],ls='dotted',color=colors[a])
        ax.plot(V_list,h_disp2[:,a,i_phi,1],ls=(0,(1,10)),color=colors[a])
        ax.plot(V_list,v_disp1[:,a,i_phi,1],ls='dashed',color=colors[a])
        ax.plot(V_list,v_disp2[:,a,i_phi,1],ls='dashdot',color=colors[a])
    ax.set_xlabel("V",fontsize=s_)
    ax.legend(handles=han[:4],labels=lab[:4],fontsize=s_)
    ax.set_ylabel("Relative weight",fontsize=s_)
    
    fig.tight_layout()
    plt.suptitle("cut "+cut+", phi="+"{:.3f}".format(phi_list[i_phi]))
    plt.show()
if 1:   #"fixed aM, function of phi"
    three = True       #Three plots
    for i_a in range(len(aM_list)):
        if three:
            fig,axs = plt.subplots(1,3)
            fig.set_size_inches(25,8)
        else:
            fig,ax = plt.subplots()
            fig.set_size_inches(10,10)
        cmap = mpl.colormaps['cool']
        
        #One
        if three:
            ax = axs[0]
        colors = cmap(np.linspace(0,1,len(V_list)))
        for v in range(0,len(V_list)):
            ax.plot(phi_list,gaps1[v,i_a,:],label="V="+"{:.3f}".format(V_list[v]),color=colors[v])
        ax.set_xlabel(r'$\phi$',fontsize=s_)
        ax.set_ylabel("gap",fontsize=s_)
        ax.legend(fontsize=s_)
        ax.set_title("Cut: "+cut,size=s_+5)
        
        if three:
            #Two
            ax = axs[1]
            ax_r = ax.twinx()
            colors = cmap(np.linspace(0,1,len(V_list)))
            for v in range(0,len(V_list)):
                ax.plot(phi_list,h_disp1[v,i_a,:,0],ls='dotted',color=colors[v],label='outer band')
                ax.plot(phi_list,h_disp2[v,i_a,:,0],ls=(0,(1,10)),color=colors[v],label='inner band')
                ax_r.plot(phi_list,v_disp1[v,i_a,:,0],ls='dashed',color=colors[v],label='lower band')
                ax_r.plot(phi_list,v_disp2[v,i_a,:,0],ls='dashdot',color=colors[v],label='higher band')
            ax.set_xlabel("phi",fontsize=s_)
            han,lab = ax.get_legend_handles_labels()
            hanr,labr = ax_r.get_legend_handles_labels()
            ax.set_ylabel('k',fontsize=s_)
            ax_r.set_ylabel('E',fontsize=s_)
            ax.set_title("Displacement",fontsize=s_)

            #Three
            ax = axs[2]
            ax.set_yticks([])
            ax_r = ax.twinx()
            colors = cmap(np.linspace(0,1,len(V_list)))
            for v in range(0,len(V_list)):
                ax_r.plot(phi_list,h_disp1[v,i_a,:,1],ls='dotted',color=colors[v])
                ax_r.plot(phi_list,h_disp2[v,i_a,:,1],ls=(0,(1,10)),color=colors[v])
                ax_r.plot(phi_list,v_disp1[v,i_a,:,1],ls='dashed',color=colors[v])
                ax_r.plot(phi_list,v_disp2[v,i_a,:,1],ls='dashdot',color=colors[v])
            ax_r.set_xlabel(r"$\phi$",fontsize=s_)
            ax_r.legend(handles=(*han[:2],*hanr[:2]),labels=(*lab[:2],*labr[:2]),fontsize=s_)
            ax_r.set_title("Relative weight",fontsize=s_)
            
            fig.tight_layout()
        plt.show()

