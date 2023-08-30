import numpy as np
import functions as fs
import matplotlib.pyplot as plt
import getopt
import sys

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:V:p:",["pts="])
    pts_ps = 150
    N = 3       #number of circles of mini-BZs
    V,phase = (7.7*1e-3, -106*np.pi/180)     #K-data
    disp = False
    save = True
    V_arr = np.linspace(0.01,0.03,3)
    phase_arr = np.linspace(0,np.pi,10)
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt in ['-V']:
        V = V_arr[int(arg)-1]
    if opt in ['-p']:
        phase = phase_arr[int(arg)-1]
    if opt == "--pts":
        pts_ps = int(arg)
if disp:
    from tqdm import tqdm
else:
    tqdm = fs.tqdm
#Savenames
dirname = "data_interlayer/"
parnames = str(N)+'_'+"{:.4f}".format(V)+'_'+"{:.4f}".format(phase)+'_'+str(pts_ps)
en_name = dirname+'energy_'+parnames+'.npy'
weight_name = dirname+'weight_'+parnames+'.npy'
lorentz_name = dirname+'lorentz_'+parnames+'.npy'
fig_name = dirname+'figure_'+parnames+'.png'

#Data
a_M = 79.8 #Angstrom 
a_mono = [3.32, 3.18]   #Angstrom, WSe2 and WS2
#Inputs
Path = 'KGC'
n_cells = int(1+3*N*(N+1))

popt_filename = "popt_interlayer.npy"
pars_H = np.load(popt_filename)
pars_V = (V,phase)

#pars_V = (0.0335,np.pi) #first in eV, second in radiants. G-data

G_M = fs.get_Moire(a_M)

path,K_points = fs.pathBZ(Path,a_mono[0],pts_ps)
try:
    res = np.load(en_name)
    weight = np.load(weight_name)
except:
    res = np.zeros((len(path),2*n_cells))
    weight = np.zeros((len(path),2*n_cells))
    for i in tqdm(range(len(path))):
        K = path[i]                                 #Considered K-point
        H = fs.big_H(K,N,pars_H,pars_V,G_M)
        res[i,:],evecs = np.linalg.eigh(H)#,subset_by_index=[n_cells_below,n_cells-1])
        for e in range(2*n_cells):
            for l in range(2):
                weight[i,e] += np.abs(evecs[n_cells*l,e])**2       ################################
    if save:
        np.save(en_name,res)
        np.save(weight_name,weight)


#Single N
en = np.zeros((len(path),2))
for i in tqdm(range(len(path))):
    H_small = fs.big_H(path[i],0,pars_H,(0,0),(0,0))
    en[i] = np.linalg.eigvalsh(H_small)
#plot points
K_list = np.linspace(-np.linalg.norm(path[0]),np.linalg.norm(path[-1]),len(path))
if 0:
    for e in range(2*n_cells):
        plt.plot(K_list,res[:,e],'k',alpha=0.2)
        for i in range(len(path)):
            if weight[i,e] > 1e-2:
               plt.scatter(K_list[i],res[i,e],s=weight[i,e],marker='o',color='r')
    plt.ylim(-1.7,-0.5)
    plt.xlim(-0.5,0.5)
    plt.show()
    exit()

#Imputs Lorentz
factor_gridy = 1
E_ = 0.03
K_ = 0.0002
larger_E = 0.2
shade_LL = 1
#Lorentzian spread
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
try:
    lor = np.load(lorentz_name)
except:
    #
    lor = np.zeros((lp,gridy))
    for i in tqdm(range(lp)):
        for j in range(2*n_cells):
            pars = (K2,E2,weight[i,j],K_list[i],res[i,j])
            lor += fs.lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
    if save:
        np.save(lorentz_name,lor)
if 1:
    from matplotlib import cm
    from matplotlib.colors import LogNorm
    #PLOTTING
    my_dpi = 1000
    px = 1/plt.rcParams['figure.dpi']/2
    fig = plt.figure(figsize=(1752*px, 2344*px))
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    #
    X,Y = np.meshgrid(K_list,E_list)
    VMIN = lor[np.nonzero(lor)].min()
    VMAX = lor.max()
    plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=VMIN, vmax=VMAX))
    plt.ylabel('eV')
    if 1:
        lll = 0.5
        plt.plot(K_list,en[:,0],'b',linewidth=lll)
        plt.plot(K_list,en[:,1],'r',linewidth=lll)
    #
    plt.ylim(-1.7,-0.5)
    plt.xlim(-0.5,0.5)

    if save:
        plt.savefig(fig_name)
    else:
        plt.show()



