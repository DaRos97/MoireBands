import numpy as np
from PIL import Image
import functions as fs
#import matplotlib.pyplot as plt
#import os
from scipy.optimize import minimize

cluster = False
#Open image of data
dirname = "/home/users/r/rossid/simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/simple_model/"
image_name = dirname + "KGK_WSe2onWS2_forDario.png"
im  = Image.open(image_name)
#Borders of image in terms of energy and momentum
E_min_fig = -1.7
E_max_fig = -0.5
K_lim = 0.5
b_up = 12   #Black border
up_size = 100   #relevant window: removed pixels from top
down_size = 900 #relevant window: removed_pixels from bottom
temp = np.array(np.asarray(im)) #use for size -> get Energy of relevant window
delta_E = (E_max_fig-E_min_fig)
E_min = E_min_fig + down_size*delta_E/temp.shape[0]
E_max = E_max_fig - up_size*delta_E/temp.shape[0]
pic = np.array(np.asarray(im)[b_up+up_size:-down_size-b_up,b_up:-b_up])
Fig_x,Fig_y,z = pic.shape
fig_x = Fig_x
fig_y = Fig_y
fac_grid_x = 20         #evaluate fig_y/2/fac_grid_x k-points per step --> 1 evaluates them all
#save new image
new_image = Image.fromarray(np.uint8(pic))
new_imagename = dirname + "cut_data.png"
new_image.save(new_imagename)
#os.system("xdg-open "+new_imagename)
#Import fitting parameters
popt_filename = dirname + "popt_interlayer.npy"
pars_H = np.load(popt_filename)

#Parameters of Moirè Hamiltonian
N = 1
a_M = 79.8 #Moirè unit length --> Angstrom 
G_M = fs.get_Moire(a_M)
a_mono = [3.32, 3.18]   #monolayer lattice length --> Angstrom, WSe2 and WS2
pts_ps = (fig_y//2)//fac_grid_x #half of k-points
path = fs.path_BZ_small(a_mono[0],pts_ps,K_lim)     #K-points in BZ cut K-G-K' with modulus between -lim and lim
K_list = np.linspace(-K_lim,K_lim,fig_y)
E_list = np.linspace(E_min,E_max,fig_x)
init_pars = [0.02,0.6,0.05,0.05]
bounds_pars = ((0.001,0.5),(0,2*np.pi),(0.03,0.06),(0.03,0.06))
minimization = True

print("Initiating minimization")
res = minimize(
        fs.image_difference,    #V,phase,E_,K_
        x0 = np.array(init_pars),
        args = (N,pic,fig_x,fig_y,fac_grid_x,E_list,K_list,pars_H,G_M,path,minimization),
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
args = (N,pic,fig_x,fig_y,fac_grid_x,E_list,K_list,pars_H,G_M,path,minimization)
final_pic = fs.image_difference(res.x,*args)

#save new image
new_image = Image.fromarray(np.uint8(final_pic))
new_imagename = dirname + "final_picture.png"
new_image.save(new_imagename)
#save minimization parameters
final_pars = np.array(res.x)
final_pars_filename = dirname + "final_pars_moire.npy"
np.save(final_pars_filename, final_pars)

print("End")
