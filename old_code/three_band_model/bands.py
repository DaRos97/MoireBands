import functions as fs
import parameters as PARS
import sys
import numpy as np
import getopt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import tqdm
import scipy.linalg as la
from time import time as tt

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "N:",["plot","LL=","UL=","path=","pts_ps=","fc","method=","EnGrid=","mono"])
    N = 1
    lower_layer = 'WSe2'
    upper_layer = 'WS2'
    Path = 'KGC'               #Points of BZ-path
    pts_ps = 50         #points per step
    plot = False
    FC = False                  #False Color plot
    method = 'GGA'
    gridy = 0
    mono = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt == '--plot':
        plot = True
    if opt == '--LL':
        lower_layer = arg
    if opt == '--UL':
        upper_layer = arg
    if opt == '--path':
        Path = arg
    if opt == '--pts_ps':
        pts_ps = int(arg)
    if opt == '--fc':
        FC = True
    if opt == '--method':
        method = arg
    if opt == '--EnGrid':
        gridy = int(arg)
    if opt == '--mono':
        mono = True

if gridy == 0:
    gridy = pts_ps*(len(Path)-1)
#
dic_sym = {'G':r'$\Gamma$', 'K':r'$K$', 'Q':r'$K/2$', 'q':r'$-K/2$', 'M':r'$M$', 'm':r'$-M$', 'N':r'$M/2$', 'n':r'$-M/2$', 'C':r'$K^\prime$', 'P':r'$K^\prime/2$', 'p':r'$-K^\prime/2$'}
print("Evaluating arpes band spectrum for "+lower_layer+"/"+upper_layer+" on path "+dic_sym[Path[0]]+"-"+dic_sym[Path[1]]+"-"+dic_sym[Path[2]]+
      "\nWith: parameters "+method+", "+str(pts_ps)+" k-points per step, "+str(N)+" circles of BZ, "+str(gridy)+" steps in E-grid.")

#Extract parameters
params_H =  [PARS.dic_params_H[method][lower_layer], PARS.dic_params_H[method][upper_layer]]
params_V =  [PARS.dic_params_V[lower_layer+'/'+upper_layer], PARS.dic_params_V[upper_layer+'/'+lower_layer]]
a_M =       PARS.dic_a_M[lower_layer+'/'+upper_layer]
offset_energy = 0.41#in eV
#####
#####Diagonalization
#####
#Here I diagonalize and obtain all the bands of the mini BZ

path,K_points = fs.pathBZ(Path,params_H[0][0],pts_ps)
data_name = "Data/res_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+".npy"
weights_name = "Data/arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+".npy"
try:    #name: LL/UL, N, Path, k-points per segment
    res = np.load(data_name)
    weight = np.load(weights_name)
    print("\nBands and ARPES weights already computed")
except:
    print("Computing bilayer energies and ARPES weights ...")
    ti = tt()
    n_cells = int(1+3*N*(N+1))*2        #dimension of H divided by 3 -> take only valence bands     #Full Diag -> *3
    sbv = [-2,0.5]                      #select_by_value for the diagonalization -> takes only bands in valence
    res = np.zeros((2,len(path),n_cells))
    weight = np.zeros((2,len(path),n_cells))
    for i in tqdm.tqdm(range(len(path))):
        K = path[i]
        H_LL = fs.total_H(K,N,params_H[0],params_V[0],a_M)     #Compute Hamiltonian for given K
        H_UL = fs.total_H(K,N,params_H[1],params_V[1],a_M)     #Compute Hamiltonian for given K
        res[0,i,:],evecs_LL = la.eigh(H_LL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
        res[1,i,:],evecs_UL = la.eigh(H_UL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
        evecs = [evecs_LL,evecs_UL]
        for l in range(2):
            for e in range(n_cells):
                for d in range(6):
                    weight[l,i,e] += np.abs(evecs[l][d,e])**2
    res[1] -= offset_energy
    np.save(data_name,res)
    np.save(weights_name,weight)
    print("Time taken: ",tt()-ti)


#########Mono-lower-layer bands
mono_LL_name = "Data/mono_"+lower_layer+'_'+Path+'_'+str(pts_ps)+".npy"
try:
    res_mono_LL = np.load(mono_LL_name)
    print("\nMono-lower-layer energies already computed")
except:
    print("\nComputing mono-lower-layer energies ...")
    ti = tt()
    res_mono_LL = np.zeros((len(path),6))
    params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
    for i in tqdm.tqdm(range(len(path))):
        K = path[i]
        H_k = fs.total_H(K,0,params_H[0],params_V,a_M)     #the only difference is in N which now is 0
        res_mono_LL[i,:],evecs_mono = np.linalg.eigh(H_k)
    np.save(mono_LL_name,res_mono_LL)
    print("Time taken: ",tt()-ti)
#########Mono-upper-layer bands
mono_UL_name = "Data/mono_"+upper_layer+'_'+Path+'_'+str(pts_ps)+".npy"
try:
    res_mono_UL = np.load(mono_UL_name)
    print("\nMono-upper-layer energies already computed")
except:
    print("\nComputing mono-upper-layer energies ...")
    ti = tt()
    res_mono_UL = np.zeros((len(path),6))
    params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
    for i in tqdm.tqdm(range(len(path))):
        K = path[i]
        H_k = fs.total_H(K,0,params_H[1],params_V,a_M)     #the only difference is in N which now is 0
        res_mono_UL[i,:],evecs_mono = np.linalg.eigh(H_k)
    np.save(mono_UL_name,res_mono_UL)
    print("Time taken: ",tt()-ti)


########
########Plot
########
if FC:          #False Color plot
    print("\nPlotting false color ...")
    ti = tt()
    bnds = len(res[0,0,:])
    #parameters of Lorentzian
    lp = len(path);     gridx = lp;    #grid in momentum fixed by points evaluated previously 
    K_ = 0.004      #spread in momentum
    K2 = K_**2
    E_ = 0.05       #spread in energy in eV
    E2 = E_**2
    min_e = np.amin(np.ravel(res[:,:bnds,:]))
    max_e = np.amax(np.ravel(res[:,:bnds,:]))
    larger_E = 0.2      #in eV. Enlargment of E axis wrt min and max band energies
    MIN_E = min_e - larger_E
    MAX_E = max_e + larger_E
    delta = MAX_E - MIN_E
    step = delta/gridy
    #K-axis
    Ki, Km, Kf = K_points
    K_list = np.linspace(-np.linalg.norm(Ki-Km),np.linalg.norm(Kf-Km),lp)
    E_list = np.linspace(MIN_E,MAX_E,gridy)
    #Compute values of lorentzian spread of weights
    lor_name = "Data/FC_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)
    par_name = '_Full_('+str(gridy)+'_'+str(larger_E).replace('.',',')+'_'+str(K_).replace('.',',')+'_'+str(E_).replace('.',',')+')'+".npy"
    lor_name += par_name
    try:
        lor = np.load(lor_name)
        print("\nLorentzian spread already computed")
    except:
        print("\nComputing Lorentzian spread ...")
        lor = np.zeros((lp,gridy))
        for l in range(2):
            for i in tqdm.tqdm(range(lp)):
                for j in range(bnds):
                    pars = (K2,E2,weight[l,i,j],K_list[i],res[l,i,j])
                    lor += fs.lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
        print("Time taken: ",tt()-ti)
        np.save(lor_name,lor)
    ## Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    if mono:
        for b in range(2):      #plot valence bands (2 for spin-rbit) of monolayer
            plt.plot(K_list,res_mono_LL[:,b],'r-',lw = 0.5)
        for b in range(2):      #plot valence bands (2 for spin-rbit) of monolayer
            plt.plot(K_list,res_mono_UL[:,b]-offset_energy,'r-',lw = 0.5)
    for i,c in enumerate([*Path]):      #plot symmetry points as vertical lines
        a = 1 if i == 2 else 0
        plt.vlines(K_list[i*lp//2-a],MIN_E,MAX_E,'k',lw=0.3,label=c)
        plt.text(K_list[i*lp//2-a],MIN_E-delta/12,dic_sym[c])
    #
    X,Y = np.meshgrid(K_list,E_list)
    plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=lor[np.nonzero(lor)].min(), vmax=lor.max()))
    plt.ylabel('eV')
    plt.show()
















#########################old code
#    sq_dimx = 100        #index range right and left
#    range_y = 2       #in eV. energy interval above and below
#    #
#    xfull = yfull = False
#    if sq_dimx*2 >= gridx:
#        sq_dimx = gridx//2
#        xfull = True
#   if range_y*2 >= delta:
#       range_y = delta/2
#       yfull = True
#   #
#   if xfull and yfull:
#       par_name = '_Full_('+str(gridy)+'_'+str(larger_E).replace('.',',')+'_'+str(K_).replace('.',',')+'_'+str(E_).replace('.',',')+')'+".npy"
#    else:
#       par_name = '_('+str(gridy)+'_'+str(larger_E).replace('.',',')+'_'+str(sq_dimx)+'_'+'{:4.3f}'.format(range_y).replace('.',',')+'_'+str(K_).replace('.',',')+'_'+str(E_).replace('.',',')+')'+".npy"
#    #
#                    continue
#                    for ix,x in enumerate(K_list):                #KKs
#                        for iy,y in enumerate(E_list):
#                            lor[ix,iy] += abs(weight[i,j])/((x-K_list[i])**2+K_**2)/((y-res[i,j])**2+E_**2)
#        else:
#            lor = np.zeros((gridx,gridy))
#            for i in tqdm.tqdm(range(lp)):
#                #get grid of X indexes around i
#                XXmin = 0 if i < sq_dimx else i-sq_dimx
#                XXmax = gridx if i+sq_dimx > gridx else i+sq_dimx
#                XX = np.linspace(XXmin,XXmax,XXmax-XXmin,endpoint=False,dtype=int)
#                for j in range(bnds):
#                   if weight[i,j] < cutoff_weight:
#                        continue
#                    #Get grid of indexes around energy res[i,j]
#                    Emin = MIN_E if res[i,j]-range_y < MIN_E else res[i,j] - range_y
#                    Emax = MAX_E if res[i,j]+range_y > MAX_E else res[i,j] + range_y
#                    ind_Emin = int((Emin-MIN_E)//step)
#                    ind_Emax = int((Emax-MIN_E)//step)
#                    YY = np.linspace(ind_Emin,ind_Emax,ind_Emax-ind_Emin,endpoint=False,dtype=int)
#                    #
#                    for ix in XX:
#                        for iy in YY:
#                            En_y = MIN_E + iy*step
#                            lor[ix,iy] += abs(weight[i,j])/((K_list[ix]-K_list[i])**2+K_**2)/((En_y-res[i,j])**2+E_**2)
#        print("Time taken: ",tt()-ti)
#        np.save(lor_name,lor)
#
#
#
#
#
#
#
#
#
#
#
#
#
#if plot:
#   print("\nPlotting bands ... ")
#   ti = tt()
#   bnds = len(res[0,:])
#   fig = plt.figure()
#   ax = fig.add_subplot(111)
#   ax.axes.get_xaxis().set_visible(False)
#   min_e = np.amin(np.ravel(res[:bnds,:]))
#   max_e = np.amax(np.ravel(res[:bnds,:]))
#   delta = abs(max_e-min_e)
#   #K values -> assuming 3 points in path
#   Ki, Km, Kf = K_points
#   K_list = np.linspace(-np.linalg.norm(Ki-Km),np.linalg.norm(Kf-Km),len(path))
#   for b in range(2):      #plot valence bands (2 for spin-rbit) of monolayer
#       plt.plot(K_list,res_mono[:,b],'r-',lw = 0.5)
#   for i,c in enumerate([*Path]):      #plot symmetry points as vertical lines
#       a = 1 if i == 2 else 0
#       plt.vlines(K_list[i*len(path)//2-a],min_e,max_e,'k',lw=0.3,label=c)
#       plt.text(K_list[i*len(path)//2-a],min_e-delta/10,dic_sym[c])
#   #real plot
#   cutoff_weight = 1e-4        #don't plot weights below this cutoff to unload the figure
#   for i in range(len(path)):
#       for j in range(bnds):
#           if weight[i,j] > cutoff_weight:
#               plt.scatter(K_list[i],res[i,j],color='b',marker='o',s=10*weight[i,j])
#    plt.ylim(-0.8,0)
#   print("Time taken: ",tt()-ti)
#   plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
