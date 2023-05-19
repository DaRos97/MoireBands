import functions as fs
import parameters as PARS
import numpy as np
import scipy.linalg as la

####not in cluster
import tqdm

def arpes_bands(args):
    N,upper_layer,lower_layer,pts_ps,Path,sbv,dirname = args
    #Internal parameters
    offset_energy = 0#.41           #NEED TO PUT CORRECT VALUE FROM FIT
    #
    hopping = [PARS.find_t(upper_layer),PARS.find_t(lower_layer)]
    epsilon = [PARS.find_e(upper_layer),PARS.find_e(lower_layer)]
    HSO = [PARS.find_HSO(upper_layer),PARS.find_HSO(lower_layer)]
    params_V =  PARS.dic_params_V[upper_layer+'/'+lower_layer]
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
    n_cells = int(1+3*N*(N+1))*14        #Dimension of H with only valence bands (14=21/3*2)
    data_name = dirname + "en_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    weights_name = dirname + "arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    try:    #name: LL/UL, N, Path, k-points per segment, select_by_value bounds
        res = np.load(data_name)
        weight = np.load(weights_name)
        print("\nBands and ARPES weights already computed")
    except:
        ti = tt()
        res = np.zeros((2,len(path),n_cells))           #Energies: 2 -> layers, len(path) -> k-points, n_cells -> dimension of Hamiltonian
        weight = np.zeros((2,len(path),n_cells))        #ARPES weights
        for i in tqdm.tqdm(range(len(path))):
            K = path[i]                                 #Considered K-point
            H_UL = fs.total_H(K,N,hopping[0],epsilon[0],HSO[0],params_V,G_M,a_mono[0])     #Compute UL Hamiltonian for given K
            H_LL = fs.total_H(K,N,hopping[1],epsilon[1],HSO[1],params_V,G_M,a_mono[1])     #Compute LL Hamiltonian for given K
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
            H_k = fs.total_H(K,0,hopping[0],epsilon[0],HSO[0],params_V,G_M,a_mono[0])     #Compute UL Hamiltonian for given K
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
            H_k = fs.total_H(K,0,hopping[1],epsilon[1],HSO[1],params_V,G_M,a_mono[1])     #Compute LL Hamiltonian for given K
            res_mono_LL[i,:],evecs_mono = la.eigh(H_k,subset_by_value=sbv)
        np.save(mono_LL_name,res_mono_LL)
        print("Time taken: ",tt()-ti)













