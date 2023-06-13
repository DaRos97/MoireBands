import numpy as np
import functions as fs
import PARAMS as PARS
import scipy.linalg as la

def grid_bands(args):
    #Parameters I need
    general_pars,grid_pars = args
    N,upper_layer,lower_layer,dirname,cluster = general_pars
    K_center, dist_kx, dist_ky, n_bands, pts_per_direction = grid_pars
    if cluster:
        tqdm = fs.tqdm
    else:
        from tqdm import tqdm
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
    weights_name_6fold = dirname + "banana_arpes_6fold_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    weights_name_3fold = dirname + "banana_arpes_3fold_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    try:    #name: LL/UL, N, K_center, grid size, number of considered valence bands
        res = np.load(data_name)
        weight_6 = np.load(weights_name_6fold)
        weight_3 = np.load(weights_name_3fold)
    except:
        print("\nComputing grid bands and ARPES weights for banana plots")
        res = np.zeros((2,pts_per_direction[0],pts_per_direction[1],n_cells-n_cells_below))           
        #Energies: 2 -> layers, grid k-pts, n_cells -> dimension of Hamiltonian
        weight_6 = np.zeros((2,pts_per_direction[0],pts_per_direction[1],n_cells-n_cells_below))        #ARPES weights
        weight_3 = np.zeros((2,pts_per_direction[0],pts_per_direction[1],n_cells-n_cells_below))        #ARPES weights
        for i in tqdm(range(pts_per_direction[0]*pts_per_direction[1])):
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
                        weight_6[l,x,y,e] += np.abs(evecs[l][d,e])**2
                        if d < 11:
                            weight_3[l,x,y,e] += np.abs(evecs[l][d,e])**2
        #Add offset energy
        res[0] += PARS.dic_params_offset[upper_layer]
        res[1] += PARS.dic_params_offset[lower_layer]
        np.save(data_name,res)
        np.save(weights_name_6fold,weight_6)
        np.save(weights_name_3fold,weight_3)

def grid_lorentz(args):
    general_pars,grid_pars,spread_pars = args
    N,upper_layer,lower_layer,dirname,cluster = general_pars
    K_center, dist_kx, dist_ky, n_bands, pts_per_direction = grid_pars
    E_cut_list, spread_Kx, spread_Ky, spread_E, fold, plot = spread_pars 
    if cluster:
        tqdm = fs.tqdm
    else:
        from tqdm import tqdm
    #
    data_name = dirname + "banana_en_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    weights_name_6fold = dirname + "banana_arpes_6fold_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    weights_name_3fold = dirname + "banana_arpes_3fold_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    res = np.load(data_name)
    if fold == "6fold":
        weight = np.load(weights_name_6fold)
    else:
        weight = np.load(weights_name_3fold)
    a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]
    #
    grid = fs.gridBZ(grid_pars,a_mono[0])
    bnds = res.shape[-1]
    Kx_list = np.linspace(-dist_kx,dist_kx,pts_per_direction[0])#grid[:,0,0]
    Ky_list = np.linspace(-dist_ky,dist_ky,pts_per_direction[1])#grid[0,:,1]
    #Compute values of lorentzian spread of weights for banana plot
    gen_lor_name = dirname + "banana_FC_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)
    lor_ = []
    for E_cut in E_cut_list:
        par_name = '_Full_'+str(spread_Kx).replace('.',',')+'_'+str(spread_E).replace('.',',')+'_E'+str(E_cut).replace('.',',')+".npy"
        lor_name = gen_lor_name + par_name
        print(lor_name)
        try:
            lor = np.load(lor_name)
        except:
            print("\nComputing banana lorentzian spread of E="+str(E_cut)+" ...")
            #lor = np.zeros((pts_per_direction[0],pts_per_direction[1]))
            lor = np.zeros((pts_per_direction[0],pts_per_direction[1]))
            Kx2 = spread_Kx**2
            Ky2 = spread_Ky**2
            E2 = spread_E**2
            L_E = 1/((res-E_cut)**2+E2)
            for i in tqdm(range(pts_per_direction[0]*pts_per_direction[1])):
                x = i%pts_per_direction[0]
                y = i//pts_per_direction[0] 
                for l in range(2):              #layers
                    for j in range(bnds):
                        pars = (Kx2,Ky2,E2,weight[l,x,y,j],res[l,x,y,j],E_cut,grid[x,y,0],grid[x,y,1])
                        lor += fs.banana_lorentzian_weight(Kx_list[:,None],Ky_list[None,:],*pars)
    #                    print(lor)
    #                    input()
            np.save(lor_name,lor)
        lor_.append(lor)

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import LogNorm
        #PLOTTING
        figname = "Figs_g/fig_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+'_'+str(spread_Kx).replace('.',',')+'_'+str(spread_E).replace('.',',')+".pdf"
        fig = plt.figure(figsize = (15,8))
        plt.suptitle("Spread K: "+str(spread_Kx)+", Spread E: "+str(spread_E))
        X,Y = np.meshgrid(Kx_list,Ky_list)
        n = len(E_cut_list)
        list_dim = [(1,1),(1,2),(2,2),(2,2),(2,3),(2,3),(3,3),(3,3)]
        fig_x, fig_y = list_dim[n-1]
        for i in range(n):
            plt.subplot(fig_x,fig_y,i+1)
            plt.title("CEM: "+str(E_cut_list[i])+" eV")
            plt.pcolormesh(X, Y,lor_[i].T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=lor_[i][np.nonzero(lor_[i])].min(), vmax=lor_[i].max()))
            plt.ylim(-0.6,0.6)
            plt.xlim(-1.5,1.5)
            plt.ylabel('Ky')
            plt.xlabel('Kx')
#        plt.savefig(figname)
        plt.show()







