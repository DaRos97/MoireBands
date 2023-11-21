import numpy as np
from PIL import Image
import K_functions as fs
import os,sys
from scipy.optimize import differential_evolution as d_e
s_ = 15
fss = (8,14)

cluster = False
###Open image of data and cut it to relevant window
home_dirname = "/home/users/r/rossid/0_simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/0_simple_model/"
dirname = home_dirname + "figs_png/"
dirname_data = home_dirname + "data_fits/"
light = "CL"    #LH,LV,CL
image_name = dirname + "cut_KK_"+light+".png"
cut_imagename = dirname + "cut_KK_"+light+"_Moire.png"

#Borders of image in terms of energy and momentum
E_min_fig = -1.4
E_max_fig = -0.2
K_min_fig = -1.1
K_max_fig = -0.1
#
im  = Image.open(image_name)
pic_0 = np.array(np.asarray(im))
len_e, len_k, z = pic_0.shape
#Cut relevant part
E_i_pix = 200
E_f_pix = len_e#len_e//2+600
K_f_pix = len_k//2 - 120
K_i_pix = 0#100
#
E_i = E_min_fig + (len_e-E_i_pix)/len_e*(E_max_fig-E_min_fig)
E_f = E_min_fig + (len_e-E_f_pix)/len_e*(E_max_fig-E_min_fig)
K_i = K_min_fig + K_i_pix/len_k*(K_max_fig-K_min_fig)
K_f = K_min_fig + K_f_pix/len_k*(K_max_fig-K_min_fig)
#
try:
    new_image = Image.open(cut_imagename)
    pic = np.array(np.asarray(new_image))
    len_e, len_k, z = pic.shape
except:
    #get Energy of relevant window
    pic = np.array(np.asarray(im)[E_i_pix:E_f_pix,K_i_pix:K_f_pix])
    len_e, len_k, z = pic.shape
    #save cut image
    new_image = Image.fromarray(np.uint8(pic))
    new_image.save(cut_imagename)
if 1: #plot cut image
    import matplotlib.pyplot as plt
    plt.figure(figsize=fss)
    plt.imshow(pic)
    plt.xticks([0,len_k//2,len_k],["{:.2f}".format(K_i),"{:.2f}".format((K_i+K_f)/2),"{:.2f}".format(K_f)])
    plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
    plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_i),"{:.2f}".format((E_i+E_f)/2),"{:.2f}".format(E_f)])
    plt.ylabel("eV",size=s_)
    plt.show()
    exit()

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
N = 5          #5-6 for cluster
k_points_factor = 1         #compute len_k//k_points_factor k-points in variational image
a_M = 79.8      #Moirè unit length --> Angstrom  #############
G_M = fs.get_Moire(a_M)     #Moirè lattice vectors
a_mono = [3.32, 3.18]       #monolayer lattice lengths --> [WSe2, WS2] Angstrom
path = fs.path_BZ_GK(a_mono[0],len_k//k_points_factor,K_f-K_i)     #K-points in BZ cut G-K with modulus between 0 and K_f-K_i
K_list = np.linspace(-(K_f-K_i),0,len_k)
E_list = np.linspace(E_f,E_i,len_e)
#Parameters and bounds for minimization
init_pars = [0.04,3.2,0.04,0.01]        #mod V, phase V, spread E, spread K
bounds_pars = ((0.001,0.1),(0,2*np.pi),(0.001,0.1),(0.001,0.1))
minimization = True
Args = (N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization)

if 1:       #Test by hand
    V = float(sys.argv[2])
    phase = float(sys.argv[3])
    VI = 0#float(sys.argv[4])
    phase_VI = 0#float(sys.argv[5])
    #VI, phi_VI = (0.0,np.pi) #interlayer Moire
    e_ = float(sys.argv[4])
    k_ = float(sys.argv[5])
    dirnamee = '/home/dario/Desktop/git/MoireBands/0_simple_model/temp_/'
    fignamee = dirnamee + 'K_'+str(N)+'_'+"{:.4f}".format(V).replace('.',',')+'_'+"{:.4f}".format(phase).replace('.',',')+'_'+"{:.4f}".format(e_).replace('.',',')+'_'+"{:.4f}".format(k_).replace('.',',')+'.npy'
    Args = (N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,False)
    par = [V,phase,e_,k_]
    import matplotlib.pyplot as plt
    plt.figure(figsize=fss)
    try:
        pic_par = np.load(fignamee)
    except:
        pic_par = fs.image_difference(par,*Args)
    s_ = 15
    plt.imshow(pic_par,cmap='gray')
    plt.text(10,80,"V="+"{:.1f}".format(V*1000)+" meV, $\phi$="+"{:.2f}".format(phase)+" rad",size = s_)
    plt.xticks([0,len_k//2,len_k],["{:.2f}".format(K_i),"{:.2f}".format((K_i+K_f)/2),"{:.2f}".format(K_f)])
    plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
    plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_i),"{:.2f}".format((E_i+E_f)/2),"{:.2f}".format(E_f)])
    plt.ylabel("eV",size=s_)
    plt.show()
    if input("Save? (y/N)")=='y':
        np.save(fignamee,pic_par)
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
