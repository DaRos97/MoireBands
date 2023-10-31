import numpy as np
import G_functions as fs
from tqdm import tqdm
import sys

#Args: 1->E_cut, 2->spread_E, 3->spread_K

cluster = True
home_dir = "/home/users/r/rossid/0_simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/0_simple_model/"
dirname = home_dir + 'data_CEM/'
N = 5
#
K_center = 'G'
dist_kx = 0.4
dist_ky = 0.4
n_pts_x = 301                #Number of k-pts in x-direction
n_pts_y = 301
pts_per_direction = (n_pts_x,n_pts_y)
grid_pars = (K_center,dist_kx,dist_ky,pts_per_direction)
#
list_E = np.linspace(-1,-0.6,21)
if cluster:
    E_cut = list_E[int(sys.argv[1])]
    spread_E = 0.01
    spread_Kx = 0.01
else:
    E_cut = float(sys.argv[1])     #eV
    spread_E = float(sys.argv[2])
    spread_Kx = spread_Ky = float(sys.argv[3])
sp = 'gauss'        #'lor'
get_spread = fs.spread_fun_dic[sp]
#
popt_filename =  home_dir + "data_fits/G_popt_interlayer.npy"
pars_H = np.load(popt_filename)

###

V = 0.02
phase = 2.6
pars_V = (V,phase)

#Moiré reciprocal lattice vectors. I start from the first one along ky and obtain the others by doing pi/3 rotations
a_mono = [3.32, 3.18]       #monolayer lattice lengths --> [WSe2, WS2] Angstrom
a_M = 79.8      #Angstrom 
G_M = fs.get_Moire(a_M)

#define k-points to compute --> use a_mono of UPPER layer
grid = fs.gridBZ(grid_pars,a_mono[0])

######################
###################### Construct Hamiltonian and compute weights in momentum grid
######################
n_cells = int(1+3*N*(N+1))        #Index of higher valence band 
data_name = dirname + "banana_en_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+".npy"
weights_name = dirname + "banana_arpes_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+".npy"
try:    #name: LL/UL, N, K_center, grid size, number of considered valence bands
    res = np.load(data_name)
    weight = np.load(weights_name)
except:
    print("\nComputing grid bands and ARPES weights for banana plots")
    res = np.zeros((pts_per_direction[0],pts_per_direction[1],2*n_cells))           
    #Energies: 2 -> layers, grid k-pts, n_cells -> dimension of Hamiltonian
    weight = np.zeros((pts_per_direction[0],pts_per_direction[1],2*n_cells))        #ARPES weights
    for i in tqdm(range(pts_per_direction[0]*pts_per_direction[1])):
        x = i%pts_per_direction[0]
        y = i//pts_per_direction[0]              
        K = grid[x,y]                                 #Considered K-point
        H_tot = fs.big_H(K,N,pars_H,pars_V,G_M,(0,0))
        res[x,y,:],evecs = np.linalg.eigh(H_tot)           #Diagonalize to get eigenvalues and eigenvectors
        for e in range(2*n_cells):
            for l in range(2):
                weight[x,y,e] += np.abs(evecs[n_cells*l,e])**2
    #Add offset energy
    np.save(data_name,res)
    np.save(weights_name,weight)

#
#   Compute CEM
#

Kx_list = np.linspace(-dist_kx,dist_kx,pts_per_direction[0])#grid[:,0,0]
Ky_list = np.linspace(-dist_ky,dist_ky,pts_per_direction[1])#grid[0,:,1]
#Compute values of lorentzian spread of weights for banana plot
gen_lor_name = dirname + "CEM_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)
par_name = '_'+str(spread_Kx).replace('.',',')+'_'+str(spread_E).replace('.',',')+'_E'+str(E_cut).replace('.',',')+".npy"
lor_name = gen_lor_name + par_name
try:
    lor = np.load(lor_name)
except:
    print("\nComputing banana lorentzian spread of E="+str(E_cut)+" ...")
    lor = np.zeros((pts_per_direction[0],pts_per_direction[1]))
    Kx2 = spread_Kx**2
    Ky2 = spread_Ky**2
    E2 = spread_E**2
    if sp=='lor':
        G_E_tot = 1/((res-E_cut)**2+E2)
    elif sp=='gauss':
        G_E_tot = np.exp(-((res-E_cut)/spread_E)**2)

    for i in tqdm(range(pts_per_direction[0]*pts_per_direction[1])):
        x = i%pts_per_direction[0]
        y = i//pts_per_direction[0] 
        G_K = get_spread(grid[x,y],spread_Kx,spread_Ky,Kx_list,Ky_list)
        for j in range(2*n_cells):
            lor += abs(weight[x,y,j]**0.5)*G_E_tot[x,y,j]*G_K

    #Normalize in color scale
    max_lor = np.max(np.ravel(lor))
    min_lor = np.min(np.ravel(np.nonzero(lor)))
    whitest = 255
    blackest = 0     
    norm_lor = np.zeros(lor.shape)
    for i in range(pts_per_direction[0]):
        for j in range(pts_per_direction[1]):
            norm_lor[i,j] = int((whitest-blackest)*(1-lor[i,j]/(max_lor-min_lor))+blackest)
    pic_lor = np.flip(norm_lor.T,axis=0)   #invert e-axis
    
    np.save(lor_name,pic_lor)

if not cluster:
    import matplotlib.pyplot as plt
    #PLOTTING
    fig = plt.figure(figsize = (15,15))
    X,Y = np.meshgrid(Kx_list,Ky_list)
    plt.gca().set_aspect('equal')
    plt.title("CEM: "+str(E_cut)+" eV")
    plt.imshow(pic_lor,cmap='gray')
#    plt.pcolormesh(X, Y,lor_[i].T,alpha=0.8,cmap=plt.cm.Greys)#,norm=LogNorm(vmin=lor_[i][np.nonzero(lor_[i])].min(), vmax=lor_[i].max()))
#            plt.ylim(-0.6,0.6)
#            plt.xlim(-1.5,1.5)
    plt.ylabel('Ky')
    plt.xlabel('Kx')
#            plt.colorbar()
#        plt.savefig(figname)
    plt.show()









