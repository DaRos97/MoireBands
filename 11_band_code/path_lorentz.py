import numpy as np
import functions as fs
import parameters as PARS

####not in cluster
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
def path_lorentz(args):
    general_pars,pars_path_bands,spread_pars_path = args
    N,upper_layer,lower_layer,dirname = general_pars
    pts_ps,Path,n_bands = pars_path_bands
    factor_gridy, E_, K_, larger_E, shade_LL, plot, plot_mono = spread_pars_path
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
        for i in tqdm.tqdm(range(lp)):
            for l in range(2):
                for j in range(bnds):
                    pars = (K2,E2,weight[l,i,j],K_list[i],res[l,i,j])
                    lor += fs.lorentzian_weight(K_list[:,None],E_list[None,:],*pars)*shade[l]
        np.save(lor_name,lor)
    if plot:
        #PLOTTING
        fig = plt.figure(figsize = (15,8))
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
        plt.show()






