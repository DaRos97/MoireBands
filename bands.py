import functions as fs
import sys
import numpy as np
import getopt
import matplotlib.pyplot as plt
from matplotlib import cm
import tqdm

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "N:",["plot","LL=","UL=","path=","pts_ps=","save","monolayer","miniBZ","bands="])
    N = 1
    lower_layer = 'WSe2'
    upper_layer = 'WS2'
    Path = 'GKMG'
    pts_ps = 20         #points per step
    plot = False
    save = False
    monolayer = False
    miniBZ = False
    BNDS = 0
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
    if opt == '--save':
        save = True
    if opt == '--monolayer':
        monolayer = True
    if opt == '--miniBZ':
        miniBZ = True
    if opt == '--bands':
        BNDS = int(arg)



#####
#####Parameters
#####
###Monolayer paerameters
#WS2 --> Table III  (first two in Angstrom, all others in eV, last in .. (lambda of SO)
dic_params_H = {'WS2':[3.191,3.144,0.717, 1.916, -0.152, -0.097, 0.590, 0.047, 0.178, 0.016, 0.069, -0.261, 0.107, -0.003, 0.109, -0.054, 0.045, 0.002, 0.325, -0.206, -0.163, 0.211],
                'WSe2':[3.325, 3.363, 0.728, 1.655, -0.146, -0.124, 0.507, 0.117, 0.127, 0.015, 0.036, -0.234, 0.107, 0.044, 0.075, -0.061, 0.032, 0.007, 0.329, -0.202, -0.164, 0.228]
            }
params_H = dic_params_H[lower_layer]
###Moirè potentials of bilayers. Different at Gamma (d_z^2 orbital) -> first two parameters, and K (d_xy orbitals) -> last two parameters
#WS2/WSe2 --> Gamma points from paper "G valley TMD moirè bands"(first in eV, second in radiants)
#WS2/WSe2 --> Louk's paper for K points(first in eV, second in radiants)
dic_params_V = {'WSe2/WS2':[0.0335,np.pi, 7.7*1e-3, -106*2*np.pi/360],
                'WS2...':[]
            }
dic_params_V['WS2/WSe2'] = dic_params_V['WSe2/WS2']
params_V = dic_params_V[lower_layer+'/'+upper_layer]
###Moirè length of bilayers in Angstrom
dic_a_M = { 'WS2/WSe2':79.8,
            'WSe2/WS2':79.8
       }
a_M = dic_a_M[lower_layer+'/'+upper_layer]

#####
#####Diagonalization
#####
#Here I diagonalize and obtain all the bands of the mini BZ
if monolayer:
    params_V = [0,0,0,0]    #no Moirè potential -> not actually needed if N=0

if miniBZ:
    a_ = a_M
else:
    a_ = params_H[0]
path = fs.pathBZ(Path,params_H[0],a_,pts_ps)
t = 'miniBZ' if miniBZ else 'BZ'
try:
    res = np.load("Data/res_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+t+".npy")
    weight = np.load("Data/arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+t+".npy")
except:
    n_cells = int(1+3*N*(N+1))*6
    res = np.zeros((n_cells,len(path)))
    weight = np.zeros((len(path),n_cells))
    for i,K in tqdm.tqdm(enumerate(path)):
        H_k = fs.total_H(K,N,params_H,params_V,a_M)
        is_Hermitian = (np.conjugate(H_k.T) == H_k).all()
        if not is_Hermitian:
            print("Error")
            exit()
        res[:,i],evecs = np.linalg.eigh(H_k)
        for e in range(len(res[:,i])//3):
            for d in range(6):
                weight[i,e] += np.abs(evecs[d,e])**2
    if save:
        np.save("Data/res_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+t+".npy",res)
        np.save("Data/arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+t+".npy",weight)

########
########Plot
########
if plot:
    bnds = len(res[:,0])
    min_bnds = bnds//3-BNDS if BNDS else 0
    max_bnds = bnds//3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    min_e = np.amin(np.ravel(res[min_bnds:max_bnds,:]))
    max_e = np.amax(np.ravel(res[min_bnds:max_bnds,:]))
    delta = abs(max_e-min_e)
    X = np.linspace(0,1,len(path))
#    for i,c in enumerate([*Path]):
#        plt.vlines(i/(len(Path)-1),min_e,max_e,'k',lw=0.3,label=c)
#        plt.text(i/(len(Path)-1),min_e-delta/10,c)
    for b in range(min_bnds,max_bnds):
        plt.plot(X,res[b,:],'k-',lw = 0.2)
    for i in range(len(path)):
        for j in range(len(res[:,i])//3):
            if weight[i,j] > 0.01 and res[j,i] > -0.8:
                plt.scatter(X[i],res[j,i],color='r',marker='o',s=weight[i,j])

    plt.ylim(-0.8,0)
    plt.show()

if 1:
    bnds = len(res[:,0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    gridx = 100
    gridy = 100
    minE = -0.8
    maxE = 0.2
    lor = np.zeros((gridx,gridy))
    X = np.linspace(0,1,len(path))
    for i in tqdm.tqdm(range(len(weight[:,0]))):
        for j in range(len(weight[0,:])//3):
            if weight[i,j] > 0.01 and res[j,i] > -0.8:
                for ix,x in enumerate(np.linspace(0,1,gridx)):
                    for iy,y in enumerate(np.linspace(minE,maxE,gridy)):
                        lor[ix,iy] += weight[i,j]/((x-X[i])**2+0.004**2)/((y-res[j,i])**2+50**2)
    Y = np.linspace(minE,maxE,gridy)
    XX,YY = np.meshgrid(X,Y)
    plt.scatter(XX,YY,c=lor,cmap=cm.get_cmap('plasma_r'))
    #plt.ylim(minE,maxE)
    plt.show()











