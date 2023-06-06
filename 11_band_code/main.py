import numpy as np
import sys
import getopt
import os

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:")
    #General parameters
    N = 2               #Number of circles of mini-BZ around the central one
    upper_layer = 'WSe2'
    lower_layer = 'WS2'
    data_dirname = "Data/"
    general_pars = (N,upper_layer,lower_layer,data_dirname)
    #Parameters for path bands
    compute_path = 0
    pts_ps = 200         #points per step
    Path = 'KGC'
    n_bands = 4         #Number of valence bands to consider
    pars_path_bands = (pts_ps,Path,n_bands)
    #Parameters for path lorentz (K-E plot)
    factor_gridy = 2        #number of points in y (energy) axis given by (len(Path)-1)*pts_ps*factor_gridy
    spread_E = 0.05
    spread_K = 1e-4      #spread in momentum
    larger_E = 0.2      #in eV. Enlargment of E axis wrt min and max band energies
    shade_LL = 1#0.5
    plot_EK = True
    plot_mono_EK = 0#True
    spread_pars_path = (factor_gridy,spread_E,spread_K,larger_E,shade_LL,plot_EK,plot_mono_EK)
    #Parameters for grid bands
    compute_grid = 1
    K_center = 'G'
    dist_kx = 1.2
    dist_ky = 0.5
    n_bands_grid = 8        #Same as n_bands above
    n_pts_x = 31                #Number of k-pts in x-direction
    n_pts_y = 31
    pts_per_direction = (n_pts_x,n_pts_y)
    grid_pars = (K_center,dist_kx,dist_ky,n_bands_grid,pts_per_direction)
    #Parameters grid lorentz (banana plot)
    E_cut = 1
    spread_Kx_banana = spread_Ky_banana = 1e-2#spread_K
    spread_E_banana = spread_E
    plot_banana = True
    spread_pars_grid = (E_cut,spread_Kx_banana,spread_Ky_banana,spread_E_banana,plot_banana)
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)

if compute_path:
    #launch path band calculation
    args_path_bands = (general_pars,pars_path_bands)
    import path_bands
    path_bands.path_bands(args_path_bands)

    #launch lorentzian spread for K-E plot
    args_path_lorentz = (general_pars,pars_path_bands,spread_pars_path)
    import path_lorentz
    path_lorentz.path_lorentz(args_path_lorentz)

if compute_grid:
    #launch grid bands for constant energy maps
    args_grid_bands = (general_pars,grid_pars)
    import grid_bands
    grid_bands.grid_bands(args_grid_bands)

    #launch banana lorentz for constant energy maps
    args_grid_lorentz = (general_pars,grid_pars,spread_pars_grid)
    import grid_lorentz
    grid_lorentz.grid_lorentz(args_grid_lorentz)






