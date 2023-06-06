import numpy as np
import functions as fs
import parameters as PARS
import scipy.linalg as la

####not in cluster
import tqdm
import matplotlib.pyplot as plt
def grid_bands(args):
    #Parameters I need
    general_pars,grid_pars = args
    N,upper_layer,lower_layer,dirname = general_pars
    K_center, dist_kx, dist_ky, n_bands, pts_per_direction = grid_pars
    # Get parameters from PARS
    hopping = [PARS.find_t(upper_layer),PARS.find_t(lower_layer)]
    epsilon = [PARS.find_e(upper_layer),PARS.find_e(lower_layer)]
    HSO = [PARS.find_HSO(upper_layer),PARS.find_HSO(lower_layer)]
    params_V =  PARS.dic_params_V[upper_layer+'/'+lower_layer]
    a_M =       PARS.dic_a_Moire[upper_layer+'/'+lower_layer]
    a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]

    #define k-points to compute --> use a_mono of UPPER layer
    grid = fs.gridBZ(grid_pars,a_mono[0])

    #Moiré reciprocal lattice vectors. I start from the first one along ky and obtain the others by doing pi/3 rotations
    G_M = fs.get_Moire(a_M)

    ######################
    ###################### Construct Hamiltonians with Moirè potential
    ######################
    n_cells = int(1+3*N*(N+1))*14        #Dimension of H with only valence bands 
    n_cells_below = int(1+3*N*(N+1))*(14-n_bands)        #Dimension of H with only valence bands - bands considered
    data_name = dirname + "banana_en_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    weights_name = dirname + "banana_arpes__"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    try:    #name: LL/UL, N, K_center, grid size, number of considered valence bands
        res = np.load(data_name)
        weight = np.load(weights_name)
    except:
        print("\nComputing grid bands and ARPES weights for banana plots")
        res = np.zeros((2,pts_per_direction[0],pts_per_direction[1],n_cells-n_cells_below))           
        #Energies: 2 -> layers, grid k-pts, n_cells -> dimension of Hamiltonian
        weight = np.zeros((2,pts_per_direction[0],pts_per_direction[1],n_cells-n_cells_below))        #ARPES weights
        for i in tqdm.tqdm(range(pts_per_direction[0]*pts_per_direction[1])):
            x = i%pts_per_direction[0]
            y = i//pts_per_direction[0]              
            K = grid[x,y]                                 #Considered K-point
            H_UL = fs.total_H(K,N,hopping[0],epsilon[0],HSO[0],params_V,G_M,a_mono[0])     #Compute UL Hamiltonian for given K
            H_LL = fs.total_H(K,N,hopping[1],epsilon[1],HSO[1],params_V,G_M,a_mono[1])     #Compute LL Hamiltonian for given K
            res[0,x,y,:],evecs_UL = la.eigh(H_UL,subset_by_index=[n_cells_below,n_cells-1])           #Diagonalize to get eigenvalues and eigenvectors
            res[1,x,y,:],evecs_LL = la.eigh(H_LL,subset_by_index=[n_cells_below,n_cells-1])           #Diagonalize to get eigenvalues and eigenvectors
            evecs = [evecs_UL,evecs_LL]
            for l in range(2):
                for e in range(n_cells-n_cells_below):
                    for d in range(22):
                        weight[l,x,y,e] += np.abs(evecs[l][d,e])**2
        #Add offset energy
        res[0] += PARS.dic_params_offset[upper_layer]
        res[1] += PARS.dic_params_offset[lower_layer]
        np.save(data_name,res)
        np.save(weights_name,weight)
    return 0
    #Plot
    plt.figure()
    plt.plot(np.arange(res.shape[1]),res[0,:,0,0]) 
    plt.show()
    exit()
