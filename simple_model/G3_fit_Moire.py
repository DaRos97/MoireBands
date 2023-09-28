import numpy as np
from PIL import Image
import G_functions as fs
import os
from scipy.optimize import differential_evolution as d_e

cluster = False
###Open image of data and cut it to relevant window
home_dirname = "/home/users/r/rossid/simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/simple_model/"
dirname = home_dirname + "figs_png/"
dirname_data = home_dirname + "data_fits/"
image_name = dirname + "cut_KGK.png"         #experimental data
im  = Image.open(image_name)
pic_0 = np.array(np.asarray(im))
len_e, len_k, z = pic_0.shape
#Borders of image in terms of energy and momentum
E_min_fig = -1.7
E_max_fig = -0.5
K_lim = 0.5
E_i = -0.55   #relevant window: energy top
E_f = -1.25   #relevant window: energy bottom
ind_Ei = len_e - int((E_min_fig-E_i)/(E_min_fig-E_max_fig)*len_e)
ind_Ef = len_e - int((E_min_fig-E_f)/(E_min_fig-E_max_fig)*len_e)
if 0: #cut to see relevant window
    new_pic = np.array(np.asarray(im)[ind_Ei:ind_Ef,:])
    new_image = Image.fromarray(np.uint8(new_pic))
    new_imagename = "temp.png"
    new_image.save(new_imagename)
    os.system("xdg-open "+new_imagename)
    exit()
#get Energy of relevant window
pic = np.array(np.asarray(im)[ind_Ei:ind_Ef,:])
len_e, len_k, z = pic.shape
#save cut image
new_image = Image.fromarray(np.uint8(pic))
new_imagename = dirname + "cut_KGK_Moire.png"
new_image.save(new_imagename)
os.system("xdg-open "+new_imagename)

if 0: #remake original image with smaller number of pixels to fit actual colors
    fig_E,fig_K,z = pic.shape
    fac_grid_K = 1         #evaluate fig_K/fac_grid_K pixels
    fac_grid_E = 1         #evaluate fig_E/fac_grid_E pixels
    len_E = fig_E//fac_grid_E
    len_K = fig_K//fac_grid_K
    new_pic = np.zeros((len_E,len_K,z))
    for i in range(len_E):
        for j in range(len_K):
            new_pic[i,j] = pic[i*fac_grid_E,j*fac_grid_K]
    if 0: #visualize reduced quality of pixels
        new_image = Image.fromarray(np.uint8(new_pic))
        new_imagename = "temp.png"
        new_image.save(new_imagename)
        os.system("xdg-open "+new_imagename)
        exit()

###Import fitting parameters
popt_filename = dirname_data + "G_popt_interlayer.npy"
pars_H = np.load(popt_filename)

###Construct the variational image
#Parameters of Moirè Hamiltonian
N = 3          #5-6 for cluster
k_points_factor = 5         #compute len_k//k_points_factor k-points in variational image
a_M = 79.8      #Moirè unit length --> Angstrom  #############
G_M = fs.get_Moire(a_M)     #Moirè lattice vectors
a_mono = [3.32, 3.18]       #monolayer lattice lengths --> [WSe2, WS2] Angstrom
path = fs.path_BZ_KGK(a_mono[0],len_k//k_points_factor,K_lim)     #K-points in BZ cut K-G-K' with modulus between -lim and lim
K_list = np.linspace(-K_lim,K_lim,len_k)
E_list = np.linspace(E_f,E_i,len_e)
#Parameters and bounds for minimization
init_pars = [0.04,3.2,0.04,0.01]        #mod V, phase V, spread E, spread K
bounds_pars = ((0.001,0.1),(0,2*np.pi),(0.001,0.1),(0.001,0.1))
minimization = True
Args = (N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization)

if 0:       #Test by hand
    list_V = [0.02,]#np.linspace(0.0235,0.0435,5)
    list_ph = [0,]#np.linspace(0,np.pi,11,endpoint=False)
    for V in list_V:
        for ph in list_ph:
            par = [0.02,0.5+1/3*np.pi,0.03,0.03]
            fs.image_difference(par,*Args)
    exit()

print("Initiating minimization")
res = d_e(
            fs.image_difference,
            x0 = np.array(init_pars),
            args = (N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization),
            bounds = bounds_pars,
            disp = False if cluster else True,
            workers = 1,
            updating = 'immediate'      #'deferred' for workers > 1
            )

print("Minimization finished with difference ",res.fun)
print("with parameters ",res.x)
print("Saving...")

minimization = False
args = (N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization)
final_pic = fs.image_difference(res.x,*args)

#save final image
new_image = Image.fromarray(np.uint8(final_pic))
new_imagename = dirname + "G_final_picture.png"
new_image.save(new_imagename)
#save minimization parameters
final_pars = np.array(res.x)
final_pars_filename = dirname_data + "G_final_pars_moire.npy"
np.save(final_pars_filename, final_pars)

print("End")
