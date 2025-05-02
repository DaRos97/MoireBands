import numpy as np
from PIL import Image
s_ = 15
import G_functions as fs
import sys
import pickle
from scipy.optimize import differential_evolution as d_e        #'stochastic'
from scipy.optimize import minimize as m_z      #'grad_descent'
type_minimization = 'grad_descent'

bands_type = 1      #can be 1->simple model, 3->three bands model with SO, 11->eleven bands model with SO
cluster = False
n_workers = 8 if (cluster and type_minimization=='stochastic') else 1
###Open image of data and cut it to relevant window
home_dirname = "/home/users/r/rossid/0_simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/0_simple_model/"
dirname = home_dirname + "figs_png/"
image_name = dirname + "cut_KGK_v1.png"         #experimental data
cut_imagename = dirname + "cut_KGK_Moire_v1.png"

#Borders of image in terms of energy and momentum
E_min_fig = -1.7
E_max_fig = -0.95#-0.5
K_lim = 0.5
E_i = E_max_fig#-0.95#-0.55   #relevant window: energy top
E_f = E_min_fig#-1.7#-1.25   #relevant window: energy bottom
try:
    new_image = Image.open(cut_imagename)
    pic = np.array(np.asarray(new_image))
    len_e, len_k, z = pic.shape
except:
    im  = Image.open(image_name)
    pic_0 = np.array(np.asarray(im))
    len_e, len_k, z = pic_0.shape
    ind_Ei = len_e - int((E_min_fig-E_i)/(E_min_fig-E_max_fig)*len_e)
    ind_Ef = len_e - int((E_min_fig-E_f)/(E_min_fig-E_max_fig)*len_e)
    #get Energy of relevant window
    pic = np.array(np.asarray(im)[ind_Ei:ind_Ef,:])
    len_e, len_k, z = pic.shape
    #save cut image
    new_image = Image.fromarray(np.uint8(pic))
    new_image.save(cut_imagename)
    #os.system("xdg-open "+new_imagename)
if 0: #plot cut image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,12))
    plt.imshow(pic)
    plt.xticks([0,len_k//2,len_k],["-0.5","0","0.5"])
    plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
    plt.yticks([0,len_e//2,len_e],["-0.55","-0.9","-1.25"])
    plt.ylabel("eV",size=s_)
    plt.show()
    exit()

#Use higher resolution -> not really needed
if 0:
    len_e *= 2
    len_k *= 2
###Import fitting parameters
dirname_data = home_dirname + "data_fits/"
if bands_type == 1:
    popt_filename = dirname_data + "G_popt_interlayer_v1.npy"
    pars_H = np.load(popt_filename)
    print(pars_H)
    exit()
elif bands_type == 3:
    filename = dirname_data + "3B_DFT_pars.pkl"
    with open(filename, 'rb') as f:
        dic_pars = pickle.load(f)
    pars_H = dic_pars['GGA']
elif bands_type == 11:
    filename_WSe2 = dirname_data + "fit_pars_WSe2_noSO.npy"
    filename_WS2 = dirname_data + "fit_pars_WS2_noSO.npy"
    pars_H = {}
    pars_H['WSe2'] = np.load(filename_WSe2)
    pars_H['WS2'] = np.load(filename_WS2)

###Construct the variational image
#Parameters of Moirè Hamiltonian
N = int(sys.argv[1])
k_points_factor = 5         #compute len_k//k_points_factor k-points in variational image
a_M = 79.8      #Moirè unit length --> Angstrom  #############
G_M = fs.get_Moire(a_M)     #Moirè lattice vectors
a_mono = [3.32, 3.18]       #monolayer lattice lengths --> [WSe2, WS2] Angstrom
path = fs.path_BZ_KGK(a_mono[0],len_k//k_points_factor,K_lim)     #K-points in BZ cut K-G-K' with modulus between -lim and lim
K_list = np.linspace(-K_lim,K_lim,len_k)
E_list = np.linspace(E_f,E_i,len_e)

if 1:       #Test by hand
    #V,phi = (0.02,np.pi)
    V = float(sys.argv[2])
    phase = float(sys.argv[3])
    e_ = float(sys.argv[4])
    k_ = float(sys.argv[5])
    dirnamee = '/home/dario/Desktop/git/MoireBands/0_simple_model/temp_/'
    fignamee = dirnamee + 'G_'+str(N)+'_'+"{:.4f}".format(V).replace('.',',')+'_'+"{:.4f}".format(phase).replace('.',',')+'_'+"{:.4f}".format(e_).replace('.',',')+'_'+"{:.4f}".format(k_).replace('.',',')+'.npy'
    Args = (N,pic,len_e,len_k,E_list,K_list,pars_H,bands_type,G_M,path,False)
    par = [V,phase,e_,k_]
    try:
        pic_par = np.load(fignamee)
    except:
        pic_par,minus = fs.image_difference(par,*Args)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,9))
    plt.imshow(pic_par,cmap='gray')
    plt.show()
    exit()
    #plt.text(len_k-260,30,"V="+"{:.1f}".format(V*1000)+" meV, $\phi$="+"{:.2f}".format(phase)+" rad",size = s_)
    plt.xticks([0,len_k//2,len_k],["-0.5","0","0.5"])
    plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
    plt.yticks([0,len_e//2,len_e],["-0.55","-0.9","-1.25"])
    plt.ylabel("eV",size=s_)
    plt.show()
    if 0:#input("Save? (y/N)")=='y':
        np.save(fignamee,pic_par)
    exit()

else:
    print("Initiating minimization")
    #Parameters and bounds for minimization
    init_pars = [0.02,np.pi,0.01,np.pi,0.02,0.02]        #mod V -> eV, phase V, spread E, spread K
    bounds_pars = ((0.001,0.1),(np.pi/2,np.pi*3/2),(0.001,0.1),(np.pi/2,np.pi*3/2),(0.01,0.06),(0.01,0.06))
    minimization = True
    args_min = (N,pic,len_e,len_k,E_list,K_list,pars_H,G_M,path,minimization)
    if type_minimization == 'stochastic':
        res = d_e(
                    fs.image_difference,
                    x0 = np.array(init_pars),
                    args = args_min,
                    bounds = bounds_pars,
                    disp = False if cluster else True,
                    workers = n_workers,
                    updating = 'immediate' if n_workers==1 else 'deferred'
                    )
    elif type_minimization == 'grad_descent':
        res = m_z(
                fs.image_difference,
                args = args_min,
                x0 = np.array(init_pars),
                bounds = bounds_pars,
                method = 'Nelder-Mead',
                options = {
                    'disp': False if cluster else True,
                    'adaptive' : True,
                    'fatol': 1e-8,
                    'xatol': 1e-8,
                    'maxiter': 1e6,
                    },
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
