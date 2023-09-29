import numpy as np
from PIL import Image
import K_functions as fs
import os
from scipy.optimize import differential_evolution as d_e

cluster = True
###Open image of data and cut it to relevant window
home_dirname = "/home/users/r/rossid/simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/simple_model/"
dirname = home_dirname + "figs_png/"
dirname_data = home_dirname + "data_fits/"
light = "LH"    #LH,LV,CL
image_name = dirname + "cut_KK_"+light+".png"
im  = Image.open(image_name)
pic_0 = np.array(np.asarray(im))
len_e, len_k, z = pic_0.shape
#Borders of image in terms of energy and momentum
E_min_fig = -1.4
E_max_fig = -0.2
K_min_fig = -1.1
K_max_fig = -0.1
#Cut relevant part
E_i_pix = 200
E_f_pix = len_e//2+200
K_f_pix = len_k//2 - 120
K_i_pix = 100
#
E_i = E_min_fig + (len_e-E_i_pix)/len_e*(E_max_fig-E_min_fig)
E_f = E_min_fig + (len_e-E_f_pix)/len_e*(E_max_fig-E_min_fig)
K_i = K_min_fig + K_i_pix/len_k*(K_max_fig-K_min_fig)
K_f = K_min_fig + K_f_pix/len_k*(K_max_fig-K_min_fig)

if 0: #cut to see relevant window
    new_pic = np.array(np.asarray(im)[E_i_pix:E_f_pix,K_i_pix:K_f_pix])
    new_image = Image.fromarray(np.uint8(new_pic))
    new_imagename = "temp.png"
    new_image.save(new_imagename)
    os.system("xdg-open "+new_imagename)
    exit()

#get Energy of relevant window
pic = np.array(np.asarray(im)[E_i_pix:E_f_pix,K_i_pix:K_f_pix])
len_e, len_k, z = pic.shape
#save cut image
new_image = Image.fromarray(np.uint8(pic))
new_imagename = dirname + "cut_KK_"+light+"_Moire.png"
new_image.save(new_imagename)
#os.system("xdg-open "+new_imagename)

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
popt_filename = dirname_data + "K_popt_interlayer.npy"
pars_H = np.load(popt_filename)

###Construct the variational image
#Parameters of Moirè Hamiltonian
N = 3          #5-6 for cluster
k_points_factor = 1         #compute len_k//k_points_factor k-points in variational image
a_M = 79.8      #Moirè unit length --> Angstrom  #############
G_M = fs.get_Moire(a_M)     #Moirè lattice vectors
a_mono = [3.32, 3.18]       #monolayer lattice lengths --> [WSe2, WS2] Angstrom
path = fs.path_BZ_GK(a_mono[0],len_k//k_points_factor,K_f-K_i)     #K-points in BZ cut K-G-K' with modulus between -lim and lim
K_list = np.linspace(-(K_f-K_i),0,len_k)
E_list = np.linspace(E_f,E_i,len_e)
#Parameters and bounds for minimization
init_pars = [0.04,3.2,0.04,0.01]        #mod V, phase V, spread E, spread K
bounds_pars = ((0.001,0.1),(0,2*np.pi),(0.001,0.1),(0.001,0.1))
minimization = True
Args = (N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization)

if 1:       #Test by hand
    import time
    list_V = np.linspace(0.01,0.06,10)
    list_ph = np.linspace(0,np.pi,10,endpoint=False)
    list_E_ = np.linspace(0.01,0.04,5)
    list_K_ = np.linspace(0.005,0.03,5)
    for V in list_V:
        for ph in list_ph:
            for e_ in list_E_:
                for k_ in list_K_:
                    par = [V,ph,e_,k_]
                    pic_par = fs.image_difference(par,*Args)
                    #
                    new_image = Image.fromarray(np.uint8(pic_par))
                    pars_name = "K_"+"{:.4f}".format(V)+'_'+"{:.4f}".format(ph)+'_'+"{:.4f}".format(e_)+'_'+"{:.4f}".format(k_)
                    new_imagename = home_dirname+"temp_image/"+pars_name+".png"
                    new_image.save(new_imagename)
                    os.system("xdg-open "+new_imagename)
    exit()

print("Initiating minimization")
n_workers = 4 if cluster else 1
res = d_e(
            fs.image_difference,
            x0 = np.array(init_pars),
            args = (N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization),
            bounds = bounds_pars,
            disp = False if cluster else True,
            workers = n_workers,
            updating = 'immediate' if n_workers==1 else 'deferred'
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
