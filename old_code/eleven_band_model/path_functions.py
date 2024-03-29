import numpy as np
import functions as fs
import PARAMS as PARS
import scipy.linalg as la

def path_bands(args):
    general_pars,pars_path_bands = args
    N,upper_layer,lower_layer,dirname,cluster = general_pars
    pts_ps,Path,n_bands = pars_path_bands
    if cluster:
        tqdm = fs.tqdm
    else:
        from tqdm import tqdm
    #Internal parameters
    hopping = [PARS.find_t(upper_layer),PARS.find_t(lower_layer)]
    epsilon = [PARS.find_e(upper_layer),PARS.find_e(lower_layer)]
    HSO = [PARS.find_HSO(upper_layer),PARS.find_HSO(lower_layer)]
    params_V =  PARS.dic_params_V[upper_layer+'/'+lower_layer]
    a_M =       PARS.dic_a_Moire[upper_layer+'/'+lower_layer]
    a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]

    #define k-points to compute --> use a_mono of UPPER layer
    path,K_points = fs.pathBZ(Path,a_mono[0],pts_ps)

    #Moiré reciprocal lattice vectors. I start from the first one and obtain the others by doing pi/3 rotations
    G_M = fs.get_Moire(a_M)

    ######################
    ###################### Construct Hamiltonians with Moirè potential
    ######################
    n_cells = int(1+3*N*(N+1))*14        #Dimension of H with only valence bands 
    n_cells_below = int(1+3*N*(N+1))*(14-n_bands)        #Dimension of H with only valence bands considered
    data_name = dirname + "en_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+".npy"
    weights_name = dirname + "arpes_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+".npy"
    try:    #name: LL/UL, N, Path, k-points per segment, number of valence bands considered
        res = np.load(data_name)
        weight = np.load(weights_name)
    except:
        print("\nComputing path bands and ARPES weights")
        res = np.zeros((2,len(path),n_cells-n_cells_below))           #Energies: 2 -> layers, len(path) -> k-points, n_cells -> dimension of Hamiltonian
        weight = np.zeros((2,len(path),n_cells-n_cells_below))        #ARPES weights
        for i in tqdm(range(len(path))):
            K = path[i]                                 #Considered K-point
            H_UL = fs.total_H(K,N,hopping[0],epsilon[0],HSO[0],params_V,G_M,a_mono[0])     #Compute UL Hamiltonian for given K
            H_LL = fs.total_H(K,N,hopping[1],epsilon[1],HSO[1],params_V,G_M,a_mono[1])     #Compute LL Hamiltonian for given K
            res[0,i,:],evecs_UL = la.eigh(H_UL,subset_by_index=[n_cells_below,n_cells-1])
            res[1,i,:],evecs_LL = la.eigh(H_LL,subset_by_index=[n_cells_below,n_cells-1])
            evecs = [evecs_UL,evecs_LL]
            for l in range(2):
                for e in range(n_cells-n_cells_below):
                    for d in range(22):
                        weight[l,i,e] += np.abs(evecs[l][d,e])**2
        #Add offset energy
        res[0] += PARS.dic_params_offset[upper_layer]
        res[1] += PARS.dic_params_offset[lower_layer]
        #Save for future use
        np.save(data_name,res)
        np.save(weights_name,weight)

    #########Mono-upper-layer bands
    mono_UL_name = dirname + "mono_"+upper_layer+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+".npy"
    try:
        res_mono_UL = np.load(mono_UL_name)
    except:
        print("\nComputing mono-upper-layer energies ...")
        res_mono_UL = np.zeros((len(path),n_bands))
        params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
        for i in range(len(path)):
            K = path[i]
            H_k = fs.total_H(K,0,hopping[0],epsilon[0],HSO[0],params_V,G_M,a_mono[0])     #Compute UL Hamiltonian for given K
            res_mono_UL[i,:],evecs_mono = la.eigh(H_k,subset_by_index=[14-n_bands,13])
        res_mono_UL += PARS.dic_params_offset[upper_layer]
        np.save(mono_UL_name,res_mono_UL)
    #########Mono-lower-layer bands
    mono_LL_name = dirname + "mono_"+lower_layer+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+".npy"
    try:
        res_mono_LL = np.load(mono_LL_name)
    except:
        print("\nComputing mono-lower-layer energies ...")
        res_mono_LL = np.zeros((len(path),n_bands))
        params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
        for i in range(len(path)):
            K = path[i]
            H_k = fs.total_H(K,0,hopping[1],epsilon[1],HSO[1],params_V,G_M,a_mono[1])     #Compute LL Hamiltonian for given K
            res_mono_LL[i,:],evecs_mono = la.eigh(H_k,subset_by_index=[14-n_bands,13])
        res_mono_LL += PARS.dic_params_offset[lower_layer]
        np.save(mono_LL_name,res_mono_LL)

def path_lorentz(args):
    general_pars,pars_path_bands,spread_pars_path = args
    N,upper_layer,lower_layer,dirname,cluster = general_pars
    pts_ps,Path,n_bands = pars_path_bands
    factor_gridy, E_, K_, larger_E, shade_LL, plot, plot_mono = spread_pars_path
    if cluster:
        tqdm = fs.tqdm
    else:
        from tqdm import tqdm
    # Preliminary steps -> import bands and weights, reconstruct K-path
    data_name = dirname + "en_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+".npy"
    weights_name = dirname + "arpes_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+".npy"
    res = np.load(data_name)
    weight = np.load(weights_name)
    a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]
    path,K_points = fs.pathBZ(Path,a_mono[0],pts_ps)
    #
    bnds = res.shape[-1]
    #parameters of Lorentzian
    lp = len(path);     gridx = lp;    #grid in momentum fixed by points evaluated previously 
    gridy = lp*factor_gridy
    K2 = K_**2
    E2 = E_**2
    min_e = np.amin(np.ravel(res))
    max_e = np.amax(np.ravel(res))
    MIN_E = min_e - larger_E
    MAX_E = max_e + larger_E
    delta = MAX_E - MIN_E
    step = delta/gridy
    #K-axis
    Ki = K_points[0]
    Kf = K_points[-1]
    K_list = np.linspace(-np.linalg.norm(Ki),np.linalg.norm(Kf),lp)
    E_list = np.linspace(MIN_E,MAX_E,gridy)
    #Compute values of lorentzian spread of weights
    lor_name = dirname + "FC_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)
    par_name = '_Full_('+str(gridy)+'_'+str(larger_E).replace('.',',')+'_'+str(K_).replace('.',',')+'_'+str(E_).replace('.',',')+'_'+str(shade_LL)+')'+".npy"
    lor_name += par_name
    try:
        lor = np.load(lor_name)
    except:
        print("\nComputing Lorentzian spread ...")
        lor = np.zeros((lp,gridy))
        shade = [1,shade_LL]
        for i in tqdm(range(lp)):
            for l in range(2):
                for j in range(bnds):
                    pars = (K2,E2,weight[l,i,j],K_list[i],res[l,i,j])
                    lor += fs.lorentzian_weight(K_list[:,None],E_list[None,:],*pars)*shade[l]
        np.save(lor_name,lor)
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import LogNorm
        #PLOTTING
        fig = plt.figure(figsize = (15,8))
        figname = "Figs_p/fig_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+'('+str(gridy)+'_'+str(larger_E).replace('.',',')+'_'+str(K_).replace('.',',')+'_'+str(E_).replace('.',',')+'_'+str(shade_LL)+').pdf'
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        if plot_mono:
            mono_UL_name = dirname + "mono_"+upper_layer+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+".npy"
            mono_LL_name = dirname + "mono_"+lower_layer+'_'+Path+'_'+str(pts_ps)+'_'+str(n_bands)+".npy"
            res_mono_UL = np.load(mono_UL_name)
            res_mono_LL = np.load(mono_LL_name)
            for b in range(n_bands):#len(res_mono_LL[0])):      #plot valence bands (2 for spin-orbit) of monolayer
                plt.plot(K_list,res_mono_LL[:,b],'g-',lw = 0.5,label=lower_layer)
            for b in range(n_bands):#len(res_mono_UL[0])):      #plot valence bands (2 for spin-orbit) of monolayer
                plt.plot(K_list,res_mono_UL[:,b],'r-',lw = 0.5,label=upper_layer)
            plt.legend()
        dic_sym = {'G':r'$\Gamma$', 'K':r'$K$', 'Q':r'$K/2$', 'q':r'$-K/2$', 'M':r'$M$', 'm':r'$-M$', 'N':r'$M/2$', 'n':r'$-M/2$', 'C':r'$K^\prime$', 'P':r'$K^\prime/2$', 'p':r'$-K^\prime/2$'}
        for i,c in enumerate([*Path]):      #plot symmetry points as vertical lines
            a = 1 if i == len(Path)-1 else 0
            plt.vlines(K_list[i*lp//(len(Path)-1)-a],MIN_E,MAX_E,'k',lw=0.3,label=c)
            plt.text(K_list[i*lp//(len(Path)-1)-a],MIN_E-delta/12,dic_sym[c])
        #
        X,Y = np.meshgrid(K_list,E_list)
        if 0:       #plot single bands -> a lot
            for i in range(bnds):
                plt.plot(K_list,res[0,:,i],'r')
                plt.plot(K_list,res[1,:,i],'g')
            plt.show()
        VMIN = lor[np.nonzero(lor)].min()
        VMAX = lor.max()
        plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=VMIN, vmax=VMAX))
        plt.ylabel('eV')
        plt.ylim(-2,MAX_E)
        plt.savefig(figname)
#        plt.show()






