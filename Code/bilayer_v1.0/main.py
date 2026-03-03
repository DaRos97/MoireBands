"""
Here we compute the final image with moiré replicas. There are a lot of parameters that enter this image.
For the monolayer parameters we take either the 'DFT' or the 'fit' values.
Details of the interlayer coupling are in the slides: we use the p_z and d_z^2 interlayer described each by w1 and w2 representing the first two moirè expansion coefficients.
We can use this script to comapare images with different interlayer w1_p and w1_d parameters -> fit "by eye".
For the moiré potential we take different values of amplitue and phase at Gamma and K.
We consider different twist angles between the layers to account for the two samples.
We take N circles of mini-BZs around the central one.

This sript is mainly for the final image and to estimate by eye the interlayer coupling.
"""
import sys,os
import numpy as np
import scipy
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_moire as fsm
from pathlib import Path
import matplotlib.pyplot as plt
machine = cfs.get_machine(os.getcwd())

if len(sys.argv)!=2:
    print("Usage: python3 main.py arg1")
    print("arg1: int -> index of parameters (see functions_moire.py)")
    quit()
else:
    ind_pars = int(sys.argv[1])  #index of parameters
    if machine == 'maf':
        ind_pars -= 1

#Parameters
monolayer_type, stacking, w2p, w2d, Vg, Vk, phiG, phiK, sample, nShells, cut, kPts = fsm.get_pars(ind_pars)
theta = 2.8 if sample=='S11' else 1.8    #twist angle, in degrees

Np = 101
Nd = 51
lW1p = np.linspace(-2.,0,Np)
lW1d = np.linspace(0,1.,Nd)
for i in range(Np*Nd):
    if i!=0 and 0:
        quit()
    inp = i//Nd
    ind = i%Nd
    w1p = lW1p[inp]#cfs.w1p_dic[monolayer_type][sample]
    w1d = lW1d[ind]#cfs.w1d_dic[monolayer_type][sample]
    print("w1p: %.4f and w1d: %.4f"%(w1p,w1d))
    parsInterlayer = {'stacking':stacking,'w1p':w1p,'w2p':w2p,'w1d':w1d,'w2d':w2d,}

    #Momentum cut values
    kList = cfs.get_kList(cut,kPts)
    kPts = kList.shape[0]

    # Preamble
    disp = 0#True
    plot_rk = 0#True     #plot real and momentum space pictures
    save_plot_rk = 0#True
    # Energy
    save_data_energy = 0#1
    plot_superimposed = 1
    show_superimposed = 0
    save_fig_superimposed = 1
    # Spread
    save_data_spread = 0
    plot_spread = 0#1
    show_spread = 0#1
    save_fig_spread = 0#1
    save_spread_txt = 0
    #
    if disp:    #print what parameters we're using
        print("-----------PARAMETRS CHOSEN-----------")
        print("Monolayers' tight-binding parameters: ",monolayer_type)
        print("Interlayer coupling: ",parsInterlayer)
        print("Sample ",sample," which has twist ",theta,"° and moiré length: "+"{:.4f}".format(cfs.moire_length(theta/180*np.pi))+" A")
        print("Moiré potential values (eV,deg): G->("+"{:.4f}".format(Vg)+","+"{:.1f}".format(phiG/np.pi*180)+"°), K->("
              +"{:.4f}".format(Vk)+","+"{:.1f}".format(phiK/np.pi*180)+"°)")
        print("Number of mini-BZs circles: ",nShells)
        print("Computing over BZ cut: ",cut," with ",kPts," points")

    if plot_rk:     #Plot real and momentum space lattices to visualize system
        fsm.plot_rk(theta,kList,cut,save_plot_rk)
        exit()

    """ Diagonalize Hamiltonian to get eigenvalues and eigenvectors at Gamma to check energies"""
    nCells = int(1+3*nShells*(nShells+1))
    args = (nShells, nCells, np.array([np.zeros(2),]), monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', save_data_energy,disp)
    evals, evecs = fsm.diagonalize_matrix(*args)

    d1 = abs(evals[0,27]+0.68)
    d2 = abs(evals[0,25]+1.34)
    if (d1<0.05 and d2<0.01) or 0:
        print("right range")

        """ Diagonalize Hamiltonian to get eigenvalues and eigenvectors """
        nCells = int(1+3*nShells*(nShells+1))
        args_fn = (monolayer_type, parsInterlayer, Vg, Vk, phiG, phiK, theta, sample, nShells, cut, kPts)
#        energy_fn = fsm.get_data_dn(machine) + 'energy_' + fsm.get_fn(*args_fn) + '.npz'
#        if not Path(energy_fn).is_file():
        args = (nShells, nCells, kList, monolayer_type, parsInterlayer, theta, (Vg,Vk,phiG,phiK), '', save_data_energy,True)
        evals,evecs = fsm.diagonalize_matrix(*args)
#        else:
#            evals = np.load(energy_fn)['evals']
#            evecs = np.load(energy_fn)['evecs']

        """ Compute bands' weights """
        weights = np.zeros((kPts,nCells*44))
        for i in range(kPts):
            ab = np.absolute(evecs[i])**2
            weights[i,:] = np.sum(ab[:22,:],axis=0) + np.sum(ab[22*nCells:22*(1+nCells),:],axis=0)

        """ Plot superimposed """
        if plot_superimposed:
            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot()
            E_max, E_min, pKi, pKf, pEmax, pEmin = cfs.dic_pars_samples[sample]
            if cut in ['Kp-G-K','Kp-G'] and 1:   #plot experimental image underneath
                from PIL import Image
                fig_fn = fsm.get_inputs_dn(machine) + sample + '_KGK.png'
                pic_raw = np.array(np.asarray(Image.open(fig_fn)))
                totPe,totPk,_ = pic_raw.shape
                K0 = np.linalg.norm(kList[0])   #val of |K|
        #        K0 = 1.4
                pK0 = int((pKf+pKi)/2)   #pixel of middle -> k=0
                pKF = int((pKf-pK0)*K0+pK0)   #pixel of k=|K|
                pKI = 2*pK0-pKF             #pixel of k=-|K|
                pic = pic_raw[pEmax:pEmin,pKI:pKF]
                if 0:   #plot full image to check bounds
                    ax.imshow(pic_raw)
                    ax.plot([pKi,pKi],[0,totPe],color='k')
                    ax.plot([pKf,pKf],[0,totPe],color='k')
                    ax.plot([pK0,pK0],[0,totPe],color='b')
                    ax.plot([pKI,pKI],[0,totPe],color='r')
                    ax.plot([pKF,pKF],[0,totPe],color='r')
                    plt.show()
                    exit()
                ax.imshow(pic,zorder=0)
                pe,pk,z = pic.shape
            else:
                pe,pk = (1000,kPts)
            if cut=='Kp-G':
                evals = np.concatenate([evals,(evals[::-1,:])[:,:]],axis=0)
                weights = np.concatenate([weights,(weights[::-1,:])[:,:]],axis=0)
                kPtsPlot = evals.shape[0]
            else:
                kPtsPlot = kPts
            kLine = np.arange(kPtsPlot)/kPtsPlot * pk
            if sample=='S11':
                evals -= 0.47
            for n in range(16*nCells,28*nCells):
                ax.plot(kLine,
                        (E_max-evals[:,n])/(E_max-E_min)*pe,
                        color='r',lw=0.1,zorder=1)
                ax.scatter(kLine,
                        (E_max-evals[:,n])/(E_max-E_min)*pe,
                        s=weights[:,n]*50,
                        lw=0,color='b',zorder=3
                        )
            ax.set_ylim(pe-1,0)
            ax.set_xlim(0,pk-1)
            ax.set_title(fsm.get_fn(*args_fn))

            if save_fig_superimposed:
                figname = fsm.get_figures_dn(machine) + "Int/superimposed_" + fsm.get_fn(*args_fn) + '.png'
                plt.savefig(figname)
            if show_superimposed:
                plt.show()
            plt.close()



exit()






n_E = 5     #number of y-ticks in image
s_ = 20     #fontsize
xticks = []
xticks_labels = cut.split('-')
for i in range(len(cut.split('-'))):
    xticks.append(kPts//(len(cut.split('-'))-1)*i)

##########################################################################
#Spread image
##########################################################################
pars_spread = (
    1e-3,       #spread K
    3e-2,       #spread E,
    'Gauss',    #type of spread
    0.01,       #delta energy
    -3 if sample=='S11' else -2.5,  #minimum energy
    -0.5 if sample=='S11' else 0    #maximum energy
)
spread_fn = fsm.get_data_dn(machine) + fsm.get_fn(*fsm.get_pars(ind_pars)) + fsm.get_fn(*pars_spread) + '.npy'
spread_fig_fn = fsm.get_figures_dn(machine) + fsm.get_fn(*fsm.get_pars(ind_pars)) + fsm.get_fn(*pars_spread) + '.png'

if disp:
    print("Computing spreading image with paramaters:")
    print("Spread function: ",pars_spread[2])
    print("Spread in K: ","{:.5f}".format(pars_spread[0])," 1/a")
    print("Spread in E: ","{:.5f}".format(pars_spread[1])," eV")

E_list = np.linspace(E_min,E_max,int((E_max-E_min)/deltaE))
if not Path(spread_fn).is_file():
    print("Computing spreading...")
    spread = np.zeros((k_pts,len(E_list)))
    for i in tqdm(range(k_pts)):
        for n in range(pars_moire[1]*15,pars_moire[1]*28):
            spread += fsm.weight_spreading(weights[i,n],K_list[i],energies[i,n],K_list,E_list[None,:],pars_spread[:3])
    if save_spread:
        np.save(spread_fn,spread)
else:
    spread = np.load(spread_fn)

if plot_spread:
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()
    spread /= np.max(spread)        #0 to 1
    map_ = 'gray_r'
    #
    ax.imshow(spread.T[::-1,:]**0.5,
              cmap=map_,
              aspect=k_pts/len(E_list),
#              aspect='auto',
              interpolation='none'
             )
    #
    ax.set_xticks(xticks,xticks_labels,size=s_)
#    ax.set_xlim

    ax.set_ylabel("energy (eV)",size=s_)
    ax.set_yticks(list(np.linspace(len(E_list),0,n_E)),
                  ["{:.2f}".format(np.linspace(E_min,E_max,n_E)[i])for i in range(n_E)],
                 size=s_)
    fig.tight_layout()
    if save_fig_spread:
        fig.savefig(spread_fig_fn)
    if show_spread:
        plt.show()
    plt.close()

if save_spread_txt:
    print("Saving in txt format")
    fn = spread_fn[:-4]+'.txt'
    if not Path(fn).is_file():
        with open(fn,'w') as f:
            for k in range(k_pts):
                for e in range(len(E_list)):
                    f.write("{:.4f}".format(K_list[k,0])+','+"{:.4f}".format(K_list[k,1])+','+"{:.4f}".format(E_list[e])+","+"{:.7f}".format(spread[k,e])+'\n')




