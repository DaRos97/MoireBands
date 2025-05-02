"""
Here we try to extract the moire potential at Gamma or K by looking at the distance between the main band and
the X shape of the side bands below.
This analysis is wrt S11, so at theta=2.8°.
We do this by only varying the Vg from 0.005 to 0.02, computing at k=Gamma (K) the distance between the main band
and the lower side band (need to get the weights, at Gamma (K) there is only one that remains nonzero). Finally,
we plot this distance as function of Vg and compare with experiment.
The observed distance in S11 is ~90 meV for Gamma and ~170 meV for K.
We also compare with different rotation angles to see the differences.

Then, we want to see why all the weight goes to the highest(lowest) side band at Gamma(K). So, we compute
the weight (projection) on the different mini-BZs considered.

We observe the weight going to just one of the side bands below the main band and we try to identify why by
plotting the projection over the different mini-BZs.

We also look at the distance between main band and side band at given energies below the VBM at G and K, by varying
both amplitude and phase of the moire potential.
Methods are slightly different for G and K.
"""
import sys,os
import numpy as np
import scipy
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions3 as fs3
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from time import time
from matplotlib.colors import Normalize
from tqdm import tqdm

momentum_point = 'G'
K_point = np.array([0,0]) if momentum_point=='G' else np.array([4/3*np.pi/cfs.dic_params_a_mono['WSe2'],0])
exp_distance_X = 0.09 if momentum_point=='G' else 0.17

compute_X = False   #X feature (EDC)
compute_projection = True   #projection of side bands on mini-BZs
compute_SB = False   #Side Band distance at same energy
plot_SB = False

machine = cfs.get_machine(cwd)
if machine=='loc':
    from tqdm import tqdm
else:
    tqdm = cfs.tqdm
#
list_V = np.linspace(0.0001,0.015,21) if momentum_point=='G' else np.linspace(0.0001,0.015,11)
monolayer_type, interlayer_symmetry, Vg, Vk, phiG, phiK, theta_0, sample, N, cut, k_pts, weight_exponent = fs3.get_pars(0)
print("-----------PARAMETRS CHOSEN-----------")
print("Monolayers' tight-binding parameters: ",monolayer_type)
print("Symmetry of interlayer coupling: ",interlayer_symmetry," with values from sample ",sample)
print("Moiré potential values (eV,deg): G->(?,"+"{:.1f}".format(phiG/np.pi*180)+"°), K->("
      +"{:.4f}".format(Vk)+","+"{:.1f}".format(phiK/np.pi*180)+"°)")
print("Twist angle: "+"{:.2f}".format(theta_0)+"° and moiré length: "+"{:.4f}".format(cfs.moire_length(theta_0/180*np.pi))+" A")
print("Number of mini-BZs circles: ",N)
print("Computing over BZ cut: ",cut," with ",k_pts," points")
#Monolayer parameters
pars_monolayer = fs3.import_monolayer_parameters(monolayer_type,machine)
#Interlayer parameters
pars_interlayer = [interlayer_symmetry,np.load(fs3.get_pars_interlayer_fn(sample,interlayer_symmetry,monolayer_type,machine))]

if compute_X:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    Ntheta = 7
    list_theta = np.linspace(theta_0-3,theta_0+3,Ntheta)
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0,1,Ntheta))
    for ind_th in range(Ntheta):
        theta = list_theta[ind_th]
        distances_X = np.zeros(len(list_V))
        for iV in range(len(list_V)):
            V = list_V[iV]
#            print("V_"+momentum_point+" = ","{:.4f}".format(V))
            #Moire parameters
            moire_potentials = (V,Vk,phiG,phiK) if momentum_point=='G' else (Vg,V,phiG,phiK)
            pars_moire = fs3.import_moire_parameters(N,moire_potentials,theta)
            look_up = fs3.lu_table(pars_moire[0])
            energies = np.zeros(pars_moire[1]*44)
            weights = np.zeros(pars_moire[1]*44)
            H_tot = fs3.big_H(K_point,look_up,pars_monolayer,pars_interlayer,pars_moire)
            energies,evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
            ab = np.absolute(evecs)**2
            ind_MB = 22 #index of main band of the layer
            weights = np.sum(ab[:ind_MB,:],axis=0) + np.sum(ab[ind_MB*pars_moire[1]:ind_MB*pars_moire[1]+ind_MB,:],axis=0)
            ind_LB = 26 if momentum_point=='G' else 27 #at K just half bands due to SOC which is degenerate at G
            inds = np.argsort(weights[pars_moire[1]*ind_LB:pars_moire[1]*28])
            ee = energies[pars_moire[1]*ind_LB+inds]
            distances_X[iV] = ee[-2]-ee[-3] if momentum_point=='G' else ee[-1]-ee[-2]   #again the SOC degeneracy

        ax.plot(list_V,distances_X,color=colors[ind_th])
    ax.axhline(y=exp_distance_X,color='g')
    txt_label = "$V_\Gamma$" if momentum_point=='G' else "$V_K$"
    ax.set_xlabel(txt_label+' (eV)',size=20)
    txt_k = "$\Gamma$" if momentum_point=='G' else "$K$"
    ax.set_ylabel(r"Energy distance at "+txt_k+" between main band and EDC (eV)",size=20)
    title = 'EDC at $\Gamma$' if momentum_point=='G' else 'EDC at $K$'
    ax.set_title(title,size=25)
    norm = Normalize(vmin=list_theta[0]-theta_0,vmax=list_theta[-1]-theta_0)
    sm = ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm,ax=ax)
    cbar.set_label(r"Twist angle $\theta$ (deg) from 2.8°",size=20)
    fig.tight_layout()
    plt.show()

if compute_projection:
    print("Computing projection of ",momentum_point)
    list_V = np.linspace(0.00,0.015,21)# if momentum_point=='G' else np.linspace(0.0001,0.015,11)
    theta = theta_0     #original one from inputs
    for iV in range(len(list_V)):
        V = list_V[iV]
        #Moire parameters
        moire_potentials = (V,Vk,phiG,phiK) if momentum_point=='G' else (Vg,V,phiG,phiK)
        pars_moire = fs3.import_moire_parameters(N,moire_potentials,theta)
        look_up = fs3.lu_table(pars_moire[0])
        #
        if 0:
            Nk = 101
            k_list = np.zeros((Nk,2))
            k_list[:,0] = np.linspace(-0.3,0.3,Nk)
            en_k = np.zeros((Nk,pars_moire[1]*44))
            wh_k = np.zeros((Nk,pars_moire[1]*44))
            for k in tqdm(range(Nk)):
                K_point = k_list[k]
                H_tot = fs3.big_H(K_point,look_up,pars_monolayer,pars_interlayer,pars_moire)
                en_k[k],evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
                ab = np.absolute(evecs)**2
                wh_k[k] = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*pars_moire[1]:22*pars_moire[1]+22,:],axis=0)
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot()
            for i in range(14):
                ax.plot(k_list[:,0],en_k[:,28*pars_moire[1]-i-1],color='r',lw=0.5,zorder=2)
                ax.scatter(k_list[:,0],en_k[:,28*pars_moire[1]-i-1],s=wh_k[:,28*pars_moire[1]-i-1]*200,color='b',marker='^',zorder=3)
            ax.set_ylim(-1.3,-1.1)
            ax.set_xlim(k_list[0,0],k_list[-1,0])
            plt.show()
        if 1:
#            K_point = np.array([0.3,0])
            H_tot = fs3.big_H(K_point,look_up,pars_monolayer,pars_interlayer,pars_moire)
            en_,evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
            ab = np.absolute(evecs)**2
            weights = np.zeros((7,pars_moire[1]*44))
            for i in range(weights.shape[0]):
                i_B = 22*i #initial index
                f_B = i_B+22
                weights[i] = np.sum(ab[i_B:f_B,:],axis=0) + np.sum(ab[22*pars_moire[1]+i_B:22*pars_moire[1]+f_B,:],axis=0)
            i_MB = 28*pars_moire[1]     #index of main band -> TVB
            l_B = i_MB-3#pars_moire[1]*2  #index of lowest band visible close to Gamma/K -> ths is the one getting the weigth at Gamma
            l_B = 28*pars_moire[1] - 4
            #
            fig = plt.figure(figsize=(20,10))
            plt.suptitle("Components of band "+str(i_MB-l_B-1)+" with weigth at "+momentum_point+" with V="+"{:.1f}".format(V*1000)+' meV')
            for i in range(7):
                ax = fig.add_subplot(2,4,i+1)
                ii = i*22
                ax.scatter(np.arange(11),ab[ii:ii+11,l_B]+ab[ii:ii+11,l_B+1],marker='o',color='b')      #add SOC degenerate band
                ax.scatter(11+np.arange(11),ab[ii+11:ii+22,l_B]+ab[ii+11:ii+22,l_B+1],marker='o',color='r')
                ax.set_title("mini BZ #"+str(i)+', total weight: '+"{:.6f}".format(np.sum(ab[ii:ii+22,l_B])))
            ax = fig.add_subplot(2,4,8)
            ax.axis('off')
            txt_list = ["d_xz","d_yz","p_z(odd)","p_x(odd)","p_y(odd)","d_z2","d_xy","d_x2-y2","p_z(even)","p_x(even)","p_y(even)"]
            for i in range(11):
                ax.text(0,1-i*0.08,str(i)+': '+txt_list[i],size=10)
            ax.text(0.5,1,"spin up",size=10,color='b')
            ax.text(0.5,0.92,"spin down",size=10,color='r')
            ax.text(0.5,0.5,"lower band weight: "+"{:.4f}".format(np.sum(ab[7*22:,l_B])),size=10,color='r')
            if 0:
                plt.savefig("results/figures/projection_"+momentum_point+"_"+"{:.1f}".format(V*1000)+'meV.png')
                plt.close()
            plt.show()

##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


if compute_SB and momentum_point=='G':
    distances_SB = np.zeros(len(list_V))
    E_reference = -1.25
    k_pts = 50
    K_list = cfs.get_K('Kp-G',2*k_pts)[k_pts:]
    print("Delta k: ",np.linalg.norm(K_list[0]-K_list[1]))
    #Extract ARPES bounds
    exp_fn = 'inputs/S11_KGK_zoom.png' if sample=='S11' else 'inputs/S3_KGK.png'
    sname = 'S11zoom' if sample=='S11' else sample
    E_max,E_min = cfs.dic_energy_bounds[sname]
    bounds = [-2*np.linalg.norm(K_list[0]),2*np.linalg.norm(K_list[0]),E_max,E_min]
    #
    pic = fs3.extract_png(exp_fn,bounds,sname)
    pe,pk,z = pic.shape
    momenta = np.linspace(bounds[0],bounds[1],pk)
    ind_E_reference = int((E_max-E_reference)/(E_max-E_min)*pe)
    #indices for E_reference = -1.35 eV
#    ind_MB = 416
#    ind_SB1 = 274
#    ind_SB2 = 310
    #indices for E_reference = -1.25 eV
    ind_MB = 490
    ind_SB1 = 328
    ind_SB2 = 390
#    ind_MB,ind_SB1,ind_SB2 = 
    #
    E_ARPES_max = abs(momenta[pk//4+ind_SB1]-momenta[pk//4+ind_MB])
    E_ARPES_min = abs(momenta[pk//4+ind_SB2]-momenta[pk//4+ind_MB])
    #
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(121)
    ax.imshow(pic[:,pk//4:-pk//4])
    ax.axhline(ind_E_reference,color='g')
    ax.scatter(ind_MB,ind_E_reference,color='r',s=20,zorder=10)
    ax.plot([ind_SB1,ind_SB2],[ind_E_reference,ind_E_reference],color='y',lw=3,zorder=10)
    ax.scatter((ind_SB1+ind_SB2)/2,ind_E_reference,color='orange',s=20,zorder=10)
    ax.set_yticks([0,ind_E_reference,pe],["{:.3f}".format(E_max),"{:.3f}".format(E_reference),"{:.3f}".format(E_min)],size=30)
    ax.set_xticks([pk//4,],["$\Gamma$",],size=30)
    ax.set_ylabel("Energy (eV)",size=30)
    ax.set_title(sample,size=30)
    #
    ax = fig.add_subplot(122)
    ax.plot(np.arange(pk//2),pic[ind_E_reference,pk//4:-pk//4,0],color='g')
    max_plot = np.max(pic[ind_E_reference,pk//4:-pk//4,0])
    ax.set_ylim(max_plot,0)
    ax.axvline(ind_MB,color='r',ls=(0,(5,10)))
    ax.fill_between([ind_SB1,ind_SB2],[max_plot,max_plot],[0,0],color='y',alpha=0.8)
    ax.axvline(int((ind_SB1+ind_SB2)/2),color='orange',ls=(0,(5,10)))
#    ax.set_xticks([ind_SB1,ind_SB2,ind_MB,pk//4],
#                  ['','','',"$\Gamma$"],size=30)
    ax.text(-1.1,0.1,'Main band: '+"{:.4f}".format(momenta[pk//4+ind_MB])+' $A^{-1}$'+'\nSide band: '+"{:.4f}".format(momenta[pk//4+ind_SB1])+' to '+"{:.4f}".format(momenta[pk//4+ind_SB2])+' $A^{-1}$'+'\nDistance between '+"{:.3f}".format(E_ARPES_min)+' and '+"{:.3f}".format(E_ARPES_max)+' $A^{-1}$',
            transform=ax.transAxes,size=20)
    ax.set_ylabel("Intensity",size=30)
    plt.show()
    print("distance between ",E_ARPES_max,E_ARPES_min)
    #
    for iV in range(len(list_V)):
        V = list_V[iV]
        print("V_"+momentum_point+" = ","{:.4f}".format(V))
        #Moire parameters
        moire_potentials = (V,Vk,phiG,phiK)
        pars_moire = fs3.import_moire_parameters(N,moire_potentials,theta)
        look_up = fs3.lu_table(pars_moire[0])
        energies = np.zeros((k_pts,pars_moire[1]*44))
        weights = np.zeros((k_pts,pars_moire[1]*44))
        #
        ind_MB = 22 #index of main band of the layer
        for i in tqdm(range(k_pts)):
            K_i = K_list[i]
            H_tot = fs3.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
            energies[i,:],evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
            ab = np.absolute(evecs)**2
            weights[i,:] = np.sum(ab[:ind_MB,:],axis=0) + np.sum(ab[ind_MB*pars_moire[1]:ind_MB*pars_moire[1]+ind_MB,:],axis=0)
        #distance
        MB = 28*pars_moire[1]-5 #main band  -> to check
        SB = 28*pars_moire[1]-1 #side band  -> sure
        argk_MB = np.argmin(abs(energies[:,MB]-E_reference))
        argk_SB = np.argmin(abs(energies[:,SB]-E_reference))
        distances_SB[iV] = np.linalg.norm(K_list[argk_SB]-K_list[argk_MB])
        print(distances_SB[iVg])
        #
        if plot_SB:
            fig,ax = plt.subplots()
            for e in range(10*pars_moire[1]):
                ax.plot(np.arange(k_pts),
                        energies[:,18*pars_moire[1]+e],
                        color='r',
                        lw=0.5,
                        zorder=2
                        )
                ax.scatter(np.arange(k_pts),
                        energies[:,18*pars_moire[1]+e],
                        s=weights[:,18*pars_moire[1]+e]**(weight_exponent)*50,
                        lw=0,
                        color='b',
                        zorder=3
                        )
            ax.axhline(E_reference,color='k')
            ax.set_ylabel("$E\;(eV)$",size=20)
            ax.set_ylim(-2.,-0.8)
            ax.set_xlim(0,k_pts)
            ax.scatter(argk_MB,energies[argk_MB,MB],color='g',s=200,alpha=0.8)
            ax.scatter(argk_SB,energies[argk_SB,SB],color='y',s=200,alpha=0.8)
            plt.show()
    #Figure
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(list_Vg,distances_SB,'b*-')
    ax.fill_between([list_Vg[0],list_Vg[-1]],[E_ARPES_min,E_ARPES_min],[E_ARPES_max,E_ARPES_max],color='y',alpha=0.6)
    ax.set_xlabel(r"$V_\Gamma$")
    ax.set_ylabel(r"Momentum distance between main band and side band")
    plt.show()

elif compute_SB and momentum_point=='K':
    check_SB = False
    Npk = 5
    for ind_pk in range(Npk):
        phiK = np.linspace(-80,-120,Npk)[ind_pk]/180*np.pi
        print("phi_K="+"{:.2f}".format(phiK/np.pi*180)+'°')
        distances_SB = np.zeros(len(list_V))
        markers = []
        E_VBM = 0.035      #energy at which to evaluate the distance, from VBM in eV
        k_pts = 100
        K_list = np.zeros((k_pts,2))
        K_list[:,0] = np.linspace(K_point[0]-0.2,K_point[0],k_pts)
        print("Delta k: ",np.linalg.norm(K_list[0]-K_list[1]))
        #
        arpes_distance = 0.11   #1/A^-1
        k_ARPES_max = arpes_distance+0.01
        k_ARPES_min = arpes_distance-0.01
        for iV in tqdm(range(len(list_V))):
            V = list_V[iV]
#            print("V_"+momentum_point+" = ","{:.4f}".format(V))
            #Moire parameters
            moire_potentials = (Vg,V,phiG,phiK)
            pars_moire = fs3.import_moire_parameters(N,moire_potentials,theta)
            look_up = fs3.lu_table(pars_moire[0])
            energies = np.zeros((k_pts,pars_moire[1]*44))
            weights = np.zeros((k_pts,pars_moire[1]*44))
            #
            ind_MB = 22 #index of main band of the layer
            for i in range(k_pts):
                K_i = K_list[i]
                H_tot = fs3.big_H(K_i,look_up,pars_monolayer,pars_interlayer,pars_moire)
                energies[i,:],evecs = scipy.linalg.eigh(H_tot,check_finite=False,overwrite_a=True)           #Diagonalize to get eigenvalues and eigenvectors
                ab = np.absolute(evecs)**2
                weights[i,:] = np.sum(ab[:ind_MB,:],axis=0) + np.sum(ab[ind_MB*pars_moire[1]:ind_MB*pars_moire[1]+ind_MB,:],axis=0)
            if check_SB: #Need to see the bands to check we are not in the gap
                fig = plt.figure(figsize=(5,10))
                ax = fig.add_subplot()
                spread_k,spread_E,type_spread,deltaE,E_min,E_max = (1e-3,5e-2,'Gauss',0.01,-3,-0.5)
                pe,pk = (1000,k_pts)
                for e in range(10*pars_moire[1]):
                    color = 'r'
                    ax.plot(np.arange(k_pts)/k_pts*pk,
                            (E_max-energies[:,18*pars_moire[1]+e])/(E_max-E_min)*pe,
                            color=color,
                            lw=0.05,
                            zorder=2
                            )
                    color = 'b'
                    ax.scatter(np.arange(k_pts)/k_pts*pk,
                            (E_max-energies[:,18*pars_moire[1]+e])/(E_max-E_min)*pe,
                            s=weights[:,18*pars_moire[1]+e]**(weight_exponent)*50,
                            lw=0,
                            color=color,
                            zorder=3
                            )
                ax.set_ylabel("$E\;(eV)$",size=20)
                ax.set_ylim(pe,0)

            #the distance in energy is from the VBM at K, so we need that first
            E_K = energies[-1,28*pars_moire[1]-1]
            E_ref = E_K - E_VBM
            #
            argk_Bs = np.argsort(abs(energies[:,28*pars_moire[1]-1]-E_ref))  #there should be 3, if not the value of energy is in the gap
            ind_MB = np.max(argk_Bs[:6])
            ind_SB = np.min(argk_Bs[:6])
            distances_SB[iV] = np.linalg.norm(K_list[ind_MB]-K_list[ind_SB])
            if check_SB:
                ax.hlines(pe*(E_max-E_ref)/(E_max-E_min),0,pk,color='r')
                ax.set_ylim(300,100)
                for i in range(6):
                    ax.scatter(argk_Bs[i],pe*(E_max-energies[argk_Bs[i],28*pars_moire[1]-1])/(E_max-E_min),color='r',s=100)
                plt.show()
                a = input("Correct? [Y/n]")
                if a == 'n':
                    marker = '^'
                else:
                    marker = 'o'
            else:
                marker = 'o'
            markers.append(marker)
            #
        #Figure
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(list_V,distances_SB,color='b')
        for i in range(len(list_V)):
            ax.scatter(list_V[i],distances_SB[i],color='b',marker=markers[i])
        ax.fill_between([list_V[0],list_V[-1]],[k_ARPES_min,k_ARPES_min],[k_ARPES_max,k_ARPES_max],color='y',alpha=0.6)
        ax.hlines(arpes_distance,list_V[0],list_V[-1],ls='--',color='r')
        ax.set_xlabel(r"$V_K$")
        ax.set_ylabel(r"Momentum distance between main band and side band at $\phi_K=$"+"{:.2f}".format(phiK/np.pi*180)+'°')
        plt.savefig('temp/SB_distance_'+"{:.2f}".format(phiK/np.pi*180)+'.png')
        plt.close()
#        plt.show() 
