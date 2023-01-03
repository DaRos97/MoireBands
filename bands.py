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
    opts, args = getopt.getopt(argv, "N:",["plot","LL=","UL=","path=","pts_ps=","fc","method="])
    N = 1
    lower_layer = 'WSe2'
    upper_layer = 'WS2'
    Path = 'GKMG'               #Points of BZ-path
    pts_ps = 50         #points per step
    plot = False
    FC = False                  #False Color plot
    method = 'GGA'
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

params_H =  PARS.dic_params_H[method][lower_layer]
params_V =  PARS.dic_params_V[lower_layer+'/'+upper_layer]
a_M =       PARS.dic_a_M[lower_layer+'/'+upper_layer]

#####
#####Diagonalization
#####
#Here I diagonalize and obtain all the bands of the mini BZ

path,K_points = fs.pathBZ(Path,params_H[0],pts_ps)
data_name = "Data/res_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+".npy"
weights_name = "Data/arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+".npy"
try:    #name: LL/UL, N, Path, k-points per segment
    res = np.load(data_name)
    weight = np.load(weights_name)
except:
    print("Computing bilayer energies ...")
    ti = tt()
    n_cells = int(1+3*N*(N+1))*2        #dimension of H divided by 3 -> take only valence bands     #Full Diag -> *3
    sbv = [-2,0.5]                      #select_by_value for the diagonalization -> takes only bands in valence
    res = np.zeros((len(path),n_cells))
    weight = np.zeros((len(path),n_cells))
    for i,K in tqdm.tqdm(enumerate(path)):
        H_k = fs.total_H(K,N,params_H,params_V,a_M)     #Compute Hamiltonian for given K
        res[i,:],evecs = la.eigh(H_k,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
        for e in range(n_cells):                        #Full Diag -> //3
            for d in range(6):
                weight[i,e] += np.abs(evecs[d,e])**2
    np.save(data_name,res)
    np.save(weights_name,weight)
    print("Time taken: ",tt()-ti)

#########Monolayer bands
mono_name = "Data/mono_"+lower_layer+"-"+upper_layer+'_0_'+Path+'_'+str(pts_ps)+".npy"
try:
    res_mono = np.load(mono_name)
except:
    print("\nComputing monolayer energies ...")
    ti = tt()
    res_mono = np.zeros((len(path),6))
    params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
    for i,K in tqdm.tqdm(enumerate(path)):
        H_k = fs.total_H(K,0,params_H,params_V,a_M)
        res_mono[i,:],evecs_mono = np.linalg.eigh(H_k)
    np.save(mono_name,res_mono)
    print("Time taken: ",tt()-ti)


########
########Plot
########
dic_sym = {'G':'gamma', 'K':'K', 'Q':'K/2', 'q':'-K/2', 'M':'M', 'm':'-M', 'N':'M/2', 'n':'-M/2', 'C':'K\'', 'P':'K\'/2', 'p':'-K\'/2'}
if plot:
    print("\nPlotting ... ")
    ti = tt()
    bnds = len(res[0,:])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    min_e = np.amin(np.ravel(res[:bnds,:]))
    max_e = np.amax(np.ravel(res[:bnds,:]))
    delta = abs(max_e-min_e)
    #K values -> assuming 3 points in path
    Ki, Km, Kf = K_points
    K_list = np.linspace(-np.linalg.norm(Ki-Km),np.linalg.norm(Kf-Km),len(path))
    for b in range(2):      #plot valence bands (2 for spin-rbit) of monolayer
        plt.plot(K_list,res_mono[:,b],'r-',lw = 0.5)
    for i,c in enumerate([*Path]):      #plot symmetry points as vertical lines
        a = 1 if i == 2 else 0
        plt.vlines(K_list[i*len(path)//2-a],min_e,max_e,'k',lw=0.3,label=c)
        plt.text(K_list[i*len(path)//2-a],min_e-delta/10,r'$'+dic_sym[c]+'$')
    #real plot
    cutoff_weight = 1e-4        #don't plot weights below this cutoff to unload the figure
    for i in range(len(path)):
        for j in range(bnds):
            if weight[i,j] > cutoff_weight:
                plt.scatter(K_list[i],res[i,j],color='b',marker='o',s=10*weight[i,j])
#    plt.ylim(-0.8,0)
    print("Time taken: ",tt()-ti)
    plt.show()

if FC:          #False Color plot
    print("\nPlotting False Color ...")
    ti = tt()
    bnds = len(res[0,:])
    #parameters of Lorentzian
    lp = len(path);     gridx = lp;    #grid in momentum fixed by points evaluated previously 
    gridy = 200     #this is actually free. Grid in energy axis
    K_ = 0.004      #spread in momentum
    K2 = K_**2
    E_ = 0.05       #spread in energy in eV
    E2 = E_**2
    cutoff_weight = 1e-4
    min_e = np.amin(np.ravel(res[:bnds,:]))
    max_e = np.amax(np.ravel(res[:bnds,:]))
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
    lor_name = lor_name + par_name
    try:
        lor = np.load(lor_name)
    except:
        if xfull and yfull:
            lor = np.zeros((lp,gridy))
            for i in tqdm.tqdm(range(lp)):
                for j in range(bnds):
                    if weight[i,j] < cutoff_weight:
                        continue
                    #arrays
                    pars = (K2,E2,weight[i,j],K_list[i],res[i,j])
                    lor += fs.lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
    ## Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    #
    #for b in range(2):      #plot valence bands (2 for spin-rbit) of monolayer
    #    plt.plot(K_list,res_mono[:,b],'r-',lw = 0.5)
    for i,c in enumerate([*Path]):      #plot symmetry points as vertical lines
        a = 1 if i == 2 else 0
        plt.vlines(K_list[i*lp//2-a],MIN_E,MAX_E,'k',lw=0.3,label=c)
        plt.text(K_list[i*lp//2-a],MIN_E-delta/10,r'$'+dic_sym[c]+'$')
    #
    X,Y = np.meshgrid(K_list,E_list)
    plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=lor[np.nonzero(lor)].min(), vmax=lor.max()))
    print("Time taken: ",tt()-ti)
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
