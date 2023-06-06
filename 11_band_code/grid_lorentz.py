import numpy as np
import functions as fs
import parameters as PARS

####not in cluster
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

dic_sym = {'G':r'$\Gamma$', 'K':r'$K$', 'Q':r'$K/2$', 'q':r'$-K/2$', 'M':r'$M$', 'm':r'$-M$', 'N':r'$M/2$', 'n':r'$-M/2$', 'C':r'$K^\prime$', 'P':r'$K^\prime/2$', 'p':r'$-K^\prime/2$'}
def grid_lorentz(args):
    general_pars,grid_pars,spread_pars = args
    N,upper_layer,lower_layer,dirname = general_pars
    K_center, dist_kx, dist_ky, n_bands, pts_per_direction = grid_pars
    E_cut, spread_Kx, spread_Ky, spread_E, plot = spread_pars 
    #
    data_name = dirname + "banana_en_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    weights_name = dirname + "banana_arpes__"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)+".npy"
    res = np.load(data_name)
    weight = np.load(weights_name)
    a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]
    #
    grid = fs.gridBZ(grid_pars,a_mono[0])
    bnds = res.shape[-1]
    Kx_list = grid[:,0,0]
    Ky_list = grid[0,:,1]
    #Compute values of lorentzian spread of weights for banana plot
    lor_name = dirname + "banana_FC_"+upper_layer+"-"+lower_layer+"_"+str(N)+'_'+K_center+'_'+str(dist_kx).replace('.',',')+'_'+str(dist_ky).replace('.',',')+'_'+str(pts_per_direction)+'_'+str(n_bands)
    par_name = '_Full_'+str(spread_Kx).replace('.',',')+'_'+str(spread_E).replace('.',',')+'_E'+str(E_cut).replace('.',',')+".npy"
    lor_name += par_name
    try:
        lor = np.load(lor_name)
    except:
        print("\nComputing banana lorentzian spread of E="+str(E_cut)+" ...")
        lor = np.zeros((pts_per_direction[0],pts_per_direction[1]))
        Kx2 = spread_Kx**2
        Ky2 = spread_Ky**2
        E2 = spread_E**2
        for i in tqdm.tqdm(range(pts_per_direction[0]*pts_per_direction[1])):
            x = i%pts_per_direction[0]
            y = i//pts_per_direction[0] 
            for l in range(2):              #layers
                for j in range(bnds):
                    pars = (Kx2,Ky2,E2,weight[l,x,y,j],res[l,x,y,j],E_cut,grid[x,y,0],grid[x,y,1])
                    lor += fs.banana_lorentzian_weight(Kx_list[:,None],Ky_list[None,:],*pars)
#                    print(lor)
#                    input()
        np.save(lor_name,lor)

    if plot:
        #PLOTTING
        fig = plt.figure(figsize = (15,8))
        X,Y = np.meshgrid(Kx_list,Ky_list)
        plt.pcolormesh(X, Y,lor.T,alpha=0.8,cmap=plt.cm.Greys,norm=LogNorm(vmin=lor[np.nonzero(lor)].min(), vmax=lor.max()))
        plt.ylabel('Ky')
        plt.xlabel('Kx')
        plt.show()







