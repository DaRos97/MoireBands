import functions as fs
import sys
import numpy as np
import getopt
import matplotlib.pyplot as plt

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "N:",["plot","LL=","UL=","path=","pts_ps=","save"])
    N = 1
    lower_layer = 'WS2'
    upper_layer = 'WSe2'
    Path = 'GKMCG'
    pts_ps = 20         #points per step
    plot = False
    save = False
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


#####
#####Parameters
#####
#WS2 --> Table III  (first two in Angstrom, all others in eV, last in .. (lambda of SO)
dic_params_H = {'WS2':[3.191,3.144,0.717, 1.916, -0.152, -0.097, 0.590, 0.047, 0.178, 0.016, 0.069, -0.261, 0.107, -0.003, 0.109, -0.054, 0.045, 0.002, 0.325, -0.206, -0.163, 0.211],
                'WSe2':[3.325, 3.363, 0.728, 1.655, -0.146, -0.124, 0.507, 0.117, 0.127, 0.015, 0.036, -0.234, 0.107, 0.044, 0.075, -0.061, 0.032, 0.007, 0.329, -0.202, -0.164, 0.228]
            }
params_H = dic_params_H[lower_layer]
#Moirè potentials of bilayers. Different at Gamma (d_z^2 orbital) -> first two parameters, and K (d_xy orbitals) -> last two parameters
#WS2/WSe2 --> Gamma points from paper "G valley TMD moirè bands"(first in eV, second in radiants)
#WS2/WSe2 --> Louk's paper for K points(first in eV, second in radiants)
dic_params_V = {'WS2/WSe2':[0.0335,np.pi, 7.7*1e-3, -106*2*np.pi/360, ],
                'WS2':[]
            }
params_V = dic_params_V[lower_layer+'/'+upper_layer]
#Moirè length of bilayers
dic_a_M = { 'WS2/WSe2':79.8,
            'WS2':0
       }
a_M = dic_a_M[lower_layer+'/'+upper_layer]

#####
#####Diagonalization
#####
#Here I diagonalize and obtain all the bands of the mini BZ

#a_M = params_H[0]
#params_V = [0,0]
#params_H[-1] = 0
path = fs.pathBZ(Path,a_M,pts_ps)
try:
    res = np.load("Data/res_"+lower_layer+"-"+upper_layer+"_"+str(N)+Path+".npy")
except:
    n_cells = int(1+3*N*(N+1))*6
    res = np.zeros((n_cells,len(path)))
    for i,K in enumerate(path):
        print(K)
        H_k = fs.total_H(K,N,params_H,params_V,a_M)
        is_Hermitian = (np.conjugate(H_k.T) == H_k).all()
        if not is_Hermitian:
            print("Error")
            exit()
        res[:,i] = np.linalg.eigvalsh(H_k)
    if save:
        np.save("Data/res_"+lower_layer+"-"+upper_layer+"_"+str(N)+Path+".npy",res)

if plot:
    bnds = len(res[:,0])
    min_bnds = 0
    max_bnds = bnds
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    min_e = np.amin(np.ravel(res[min_bnds:max_bnds,:]))
    max_e = np.amax(np.ravel(res[min_bnds:max_bnds,:]))
    for i,c in enumerate([*Path]):
        plt.vlines(i/(len(Path)-1),min_e,max_e,'k',lw=0.3,label=c)
        plt.text(i/(len(Path)-1),min_e-0.2,c)
    for b in range(min_bnds,max_bnds):
        X = np.linspace(0,1,len(path))
        plt.plot(X,res[b,:])

    plt.show()











