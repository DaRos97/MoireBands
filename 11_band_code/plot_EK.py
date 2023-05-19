import numpy as np
import functions as fs
import parameters as PARS
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm


def plot_EK(args):
    N,upper_layer,lower_layer,pts_ps,Path,sbv,factor_gridy,E_,K_,larger_E,plot_mono,dirname = args
    offset_energy = 0#.4
    #
    data_name = dirname + "en_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    weights_name = dirname + "arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    res = np.load(data_name)
    mono_UL_name = dirname + "mono_"+upper_layer+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    res_mono_UL = np.load(mono_UL_name)
    mono_LL_name = dirname + "mono_"+lower_layer+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    res_mono_LL = np.load(mono_LL_name)
    weight = np.load(weights_name)
    a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]
    path,K_points = fs.pathBZ(Path,a_mono[0],pts_ps)
    #
    dic_sym = {'G':r'$\Gamma$', 'K':r'$K$', 'Q':r'$K/2$', 'q':r'$-K/2$', 'M':r'$M$', 'm':r'$-M$', 'N':r'$M/2$', 'n':r'$-M/2$', 'C':r'$K^\prime$', 'P':r'$K^\prime/2$', 'p':r'$-K^\prime/2$'}
    bnds = len(res[0,0,:])
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
    #Ki, Km, Kf = K_points
    Ki = K_points[0]
    Kf = K_points[-1]
    K_list = np.linspace(-np.linalg.norm(Ki),np.linalg.norm(Kf),lp)
    E_list = np.linspace(MIN_E,MAX_E,gridy)
    #Compute values of lorentzian spread of weights
    lor_name = dirname + "FC_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)
    par_name = '_Full_('+str(gridy)+'_'+str(larger_E).replace('.',',')+'_'+str(K_).replace('.',',')+'_'+str(E_).replace('.',',')+')'+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    lor_name += par_name
    lor = np.load(lor_name)
    #
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
