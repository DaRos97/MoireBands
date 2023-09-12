import numpy as np
from PIL import Image
import functions as fs
#import matplotlib.pyplot as plt
#import os
from scipy.optimize import minimize

cluster = False
###Open image of data and cut it to relevant window
dirname = "/home/users/r/rossid/simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/simple_model/"
image_name = dirname + "KGK_WSe2onWS2_forDario.png"         #experimental data
im  = Image.open(image_name)
#Borders of image in terms of energy and momentum
E_min_fig = -1.7
E_max_fig = -0.5
K_lim = 0.5
b_up = 12   #Black border of image
up_size = 100   #relevant window: removed pixels from top
down_size = 900 #relevant window: removed pixels from bottom
#get Energy of relevant window
temp = np.array(np.asarray(im)).shape[0]    #pixels of y direction (energy) 
delta_E = (E_max_fig-E_min_fig)
E_min = E_min_fig + down_size*delta_E/temp
E_max = E_max_fig - up_size*delta_E/temp
pic = np.array(np.asarray(im)[b_up+up_size:-down_size-b_up,b_up:-b_up])
fig_E,fig_K,z = pic.shape
fac_grid_K = 10         #evaluate fig_K/2/fac_grid_x k-points per step --> 1 evaluates them all
#save cut image
new_image = Image.fromarray(np.uint8(pic))
new_imagename = dirname + "cut_data.png"
new_image.save(new_imagename)
#os.system("xdg-open "+new_imagename)
###Import fitting parameters
popt_filename = dirname + "popt_interlayer.npy"
pars_H = np.load(popt_filename)

###Construct the variational image
#Parameters of Moirè Hamiltonian
N = 3           #3 for cluster
a_M = 79.8      #Moirè unit length --> Angstrom 
G_M = fs.get_Moire(a_M)     #Moirè lattice vectors
a_mono = [3.32, 3.18]       #monolayer lattice lengths --> Angstrom, WSe2 and WS2
pts_ps = (fig_K//2)//fac_grid_K     #half of k-points
path = fs.path_BZ_small(a_mono[0],pts_ps,K_lim)     #K-points in BZ cut K-G-K' with modulus between -lim and lim
K_list = np.linspace(-K_lim,K_lim,fig_K)
E_list = np.linspace(E_min,E_max,fig_E)
#Parameters and bounds for minimization
init_pars = [0.02,0.6,0.05,0.05]        #mod V, phase V, spread E, spread K
bounds_pars = ((0.001,0.1),(0,2*np.pi),(0.001,0.1),(0.001,0.1))
minimization = True

print("Initiating minimization")
res = minimize(
        fs.image_difference,
        x0 = np.array(init_pars),
        args = (N,pic,fig_E,fig_K,fac_grid_K,E_list,K_list,pars_H,G_M,path,minimization),
        bounds = bounds_pars,
        options = {
            'disp':False if cluster else True,
#            'maxiter':1e1,
            },
#        tol = 1e-8,
        method = 'Nelder-Mead',
        )

print("Minimization finished with difference ",res.fun)
print("with parameters ",res.x)
print("Saving...")

minimization = False
args = (N,pic,fig_E,fig_K,fac_grid_K,E_list,K_list,pars_H,G_M,path,minimization)
final_pic = fs.image_difference(res.x,*args)

#save final image
new_image = Image.fromarray(np.uint8(final_pic))
new_imagename = dirname + "final_picture.png"
new_image.save(new_imagename)
#save minimization parameters
final_pars = np.array(res.x)
final_pars_filename = dirname + "final_pars_moire.npy"
np.save(final_pars_filename, final_pars)

print("End")
