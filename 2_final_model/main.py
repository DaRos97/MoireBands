import numpy as np
import sys
import getopt
import os

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["spread=","path","grid","cluster","fold="])
    #General parameters
    N = 0              #Number of circles of mini-BZ around the central one
    upper_layer = 'WSe2'
    lower_layer = 'WS2'
    data_dirname = "Data/"
    cluster = False
    #Parameters for path bands
    compute_path = False
    pts_ps = 50         #points per step
    Path = 'KGC'
    n_bands = 6         #Number of valence bands to consider
    pars_path_bands = (pts_ps,Path,n_bands)
    #Parameters for path lorentz (K-E plot)
    factor_gridy = 2        #number of points in y (energy) axis given by (len(Path)-1)*pts_ps*factor_gridy
    spread_E = 0.05
    spread_K = 1e-4      #spread in momentum
    larger_E = 0.2      #in eV. Enlargment of E axis wrt min and max band energies
    shade_LL = 1        #NOT using it
    plot_EK = True
    plot_mono_EK = True
    #Parameters for grid bands
    compute_grid = False
    K_center = 'G'
    dist_kx = 0.6
    dist_ky = 0.6
    n_bands_grid = 4        #Same as n_bands above
    n_pts_x = 151                #Number of k-pts in x-direction
    n_pts_y = 151
    pts_per_direction = (n_pts_x,n_pts_y)
    #Parameters grid lorentz (banana plot)
    E_cut = [-1.2]#[-0.8,-0.85,-0.9,-0.95,-1,-1.05,-1.1,-1.15]
    spread_ind = 0
    spread_Kx_banana = spread_Ky_banana = 0.01
    spread_E_banana = 0.05#spread_E
    fold = "6fold"
    plot_banana = True
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt == "--path":
        compute_path = True
    if opt == "--grid":
        compute_grid = True
    if opt == "--cluster":
        cluster = True
        data_dirname = "/home/users/r/rossid/1_MOIRE/Data/"
        plot_EK = False
        plot_mono_EK = False
        plot_banana = False
    if opt == "--fold":
        fold = arg + "fold"
    if opt == "--spread":   #not so good
        spread_ind = int(arg)
        list_s_k = np.logspace(-3,-10,base=2,num=10)
        list_s_E = np.logspace(-3,-10,base=2,num=10)
        #
        spread_Kx_banana = spread_Ky_banana = list_s_k[spread_ind//10]
        spread_E_banana = list_s_E[spread_ind%10]
        #
        spread_K = list_s_k[spread_ind//10]
        spread_E = list_s_E[spread_ind%10]

if 1:#Shady stuff
    ens = np.load("temp_ens.npy")
    evs = np.load("temp_evs.npy")
    n_ = [24,25,26,27]
    for n in n_:
        print("band: ",n)
        for i in range(evs.shape[0]):
            if abs(np.linalg.norm(evs[i,n]))>1e-2:
                print(i,np.linalg.norm(evs[i,n]))
    exit()

#
general_pars = (N,upper_layer,lower_layer,data_dirname,cluster)
spread_pars_path = (factor_gridy,spread_E,spread_K,larger_E,shade_LL,plot_EK,plot_mono_EK)
grid_pars = (K_center,dist_kx,dist_ky,n_bands_grid,pts_per_direction)
spread_pars_grid = (E_cut,spread_Kx_banana,spread_Ky_banana,spread_E_banana,fold,plot_banana)

if compute_path:
    import path_functions
    #launch path band calculation
    args_path_bands = (general_pars,pars_path_bands)
    path_functions.path_bands(args_path_bands)
    #launch lorentzian spread for K-E plot
    args_path_lorentz = (general_pars,pars_path_bands,spread_pars_path)
    path_functions.path_lorentz(args_path_lorentz)

if compute_grid:
    import grid_functions
    #launch grid bands for constant energy maps
    args_grid_bands = (general_pars,grid_pars)
    grid_functions.grid_bands(args_grid_bands)
    #launch banana lorentz for constant energy maps
    args_grid_lorentz = (general_pars,grid_pars,spread_pars_grid)
    grid_functions.grid_lorentz(args_grid_lorentz)






