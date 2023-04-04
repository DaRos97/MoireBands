import functions as fs
import parameters as PARS
import sys
import numpy as np
import getopt
import scipy.linalg as la
from time import time as tt

####not in cluster
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

dirname = "../Data/11_bands/"                    #WRONG
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["Path=","LL=","UL=","pts_ps=","mono","gridy=","spread_E="])
    N = 4               #Number of circles of mini-BZ around the central one
    upper_layer = 'WSe2'
    lower_layer = 'WS2'
    pts_ps = 200         #points per step
    Path = 'KGC'
    plot_mono = False
    factor_gridy = 1
    spread_E = 0.005
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt == '--LL':
        lower_layer = arg
    if opt == '--UL':
        upper_layer = arg
    if opt == '--pts_ps':
        pts_ps = int(arg)
    if opt == '--mono':
        plot_mono = True
    if opt == '--Path':
        Path = arg
    if opt == '--gridy':
        factor_gridy = int(arg)
    if opt == '--spread_E':
        spread_E = float(arg)
#Internal parameters
offset_energy = 0#.41
#
hopping = [PARS.find_t(upper_layer),PARS.find_t(lower_layer)]
epsilon = [PARS.find_e(upper_layer),PARS.find_e(lower_layer)]
HSO = [PARS.find_HSO(upper_layer),PARS.find_HSO(lower_layer)]
params_V =  [PARS.dic_params_V[upper_layer+'/'+lower_layer], PARS.dic_params_V[lower_layer+'/'+upper_layer]]
a_M =       PARS.dic_a_Moire[upper_layer+'/'+lower_layer]
a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]

#define k-points to compute --> use a_mono of UPPER layer
path,K_points = fs.pathBZ(Path,a_mono[0],pts_ps)

#Moiré reciprocal lattice vectors. I start from the first one and obtain the others by doing pi/3 rotations
G_M = [4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
for i in range(1,6):
    G_M.append(np.tensordot(fs.R_z(np.pi/3*i),G_M[0],1))

######################
###################### Construct Hamiltonians with Moirè potential
######################
n_cells = int(1+3*N*(N+1))*14        #Dimension of H divided by 3 -> take only valence bands
sbv = [-10,1]                      #Select_by_value for the diagonalization in order to take only bands in valence. 
data_name = dirname + "en_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
weights_name = dirname + "arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
try:    #name: LL/UL, N, Path, k-points per segment
    res = np.load(data_name)
    weight = np.load(weights_name)
    print("\nBands and ARPES weights already computed")
except:
    ti = tt()
    res = np.zeros((2,len(path),n_cells))           #Energies: 2 -> layers, len(path) -> k-points, n_cells -> dimension of Hamiltonian
    weight = np.zeros((2,len(path),n_cells))        #ARPES weights
    for i in tqdm.tqdm(range(len(path))):
        K = path[i]                                 #Considered K-point
        H_UL = fs.total_H(K,N,hopping[0],epsilon[0],HSO[0],params_V[0],G_M,a_mono[0])     #Compute UL Hamiltonian for given K
        H_LL = fs.total_H(K,N,hopping[1],epsilon[1],HSO[1],params_V[1],G_M,a_mono[1])     #Compute LL Hamiltonian for given K
        res[0,i,:],evecs_UL = la.eigh(H_UL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
        res[1,i,:],evecs_LL = la.eigh(H_LL,subset_by_value=sbv)           #Diagonalize to get eigenvalues and eigenvectors
        evecs = [evecs_UL,evecs_LL]
        for l in range(2):
            for e in range(n_cells):
                for d in range(22):
                    weight[l,i,e] += np.abs(evecs[l][d,e])**2
    #res[1] -= offset_energy             #remove offset energy from LOWER layer
    np.save(data_name,res)
    np.save(weights_name,weight)
    print("Time taken: ",tt()-ti)

#########Mono-upper-layer bands
mono_UL_name = dirname + "mono_"+upper_layer+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
try:
    res_mono_UL = np.load(mono_UL_name)
    print("\nMono-upper-layer energies already computed")
except:
    print("\nComputing mono-upper-layer energies ...")
    ti = tt()
    res_mono_UL = np.zeros((len(path),14))
    params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
    for i in tqdm.tqdm(range(len(path))):
        K = path[i]
        H_k = fs.total_H(K,0,hopping[0],epsilon[0],HSO[0],params_V[0],G_M,a_mono[0])     #Compute UL Hamiltonian for given K
        res_mono_UL[i,:],evecs_mono = la.eigh(H_k,subset_by_value=sbv)
    np.save(mono_UL_name,res_mono_UL)
    print("Time taken: ",tt()-ti)
#########Mono-lower-layer bands
mono_LL_name = dirname + "mono_"+lower_layer+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
try:
    res_mono_LL = np.load(mono_LL_name)
    print("\nMono-lower-layer energies already computed")
except:
    print("\nComputing mono-lower-layer energies ...")
    ti = tt()
    res_mono_LL = np.zeros((len(path),14))
    params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0
    for i in tqdm.tqdm(range(len(path))):
        K = path[i]
        H_k = fs.total_H(K,0,hopping[1],epsilon[1],HSO[1],params_V[1],G_M,a_mono[1])     #Compute LL Hamiltonian for given K
        res_mono_LL[i,:],evecs_mono = la.eigh(H_k,subset_by_value=sbv)
    np.save(mono_LL_name,res_mono_LL)
    print("Time taken: ",tt()-ti)
############

dic_sym = {'G':r'$\Gamma$', 'K':r'$K$', 'Q':r'$K/2$', 'q':r'$-K/2$', 'M':r'$M$', 'm':r'$-M$', 'N':r'$M/2$', 'n':r'$-M/2$', 'C':r'$K^\prime$', 'P':r'$K^\prime/2$', 'p':r'$-K^\prime/2$'}
print("\nPlotting false color ...")
ti = tt()
bnds = len(res[0,0,:])
#parameters of Lorentzian
lp = len(path);     gridx = lp;    #grid in momentum fixed by points evaluated previously 
gridy = lp*factor_gridy
K_ = 0.001      #spread in momentum
K2 = K_**2
E_ = spread_E#0.005       #spread in energy in eV
E2 = E_**2
min_e = np.amin(np.ravel(res))
max_e = np.amax(np.ravel(res))
larger_E = 0.2      #in eV. Enlargment of E axis wrt min and max band energies
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
try:
    lor = np.load(lor_name)
    print("\nLorentzian spread already computed")
except:
    print("\nComputing Lorentzian spread ...")
    lor = np.zeros((lp,gridy))
    for i in tqdm.tqdm(range(lp)):
        for l in range(2):
            for j in range(bnds):
                #if res[l,i,j] < MIN_E or res[l,i,j] > MAX_E:
                #    continue
                pars = (K2,E2,weight[l,i,j],K_list[i],res[l,i,j])
                lor += fs.lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
    print("Time taken: ",tt()-ti)
    np.save(lor_name,lor)
## Plot
if 1:
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




















