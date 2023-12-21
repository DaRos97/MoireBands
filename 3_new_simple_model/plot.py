import numpy as np
from scipy.optimize import minimize
from PIL import Image
import functions as fs
from pathlib import Path

home_dirname = "/home/dario/Desktop/git/MoireBands/3_new_simple_model/"
fig_dirname = home_dirname + "input_figures/"
data_dirname = home_dirname + "input_data/"
#Borders of image in terms of energy and momentum
version = "v1"
E_min_fig = -1.5
E_max_fig = -1
K_lim = 0.5
bounds_pic = (E_min_fig,E_max_fig,K_lim)
try:    #Reference image
    pic = np.load(fs.compute_picture_filename(version,bounds_pic,fig_dirname))
except:
    pic = fs.cut_image(bounds_pic,version,fig_dirname,True)
if 0: #plot image
    fs.plot_image(pic,bounds_pic)
#Import Hamiltonian parameters
Hopt = np.load(fs.compute_Hopt_filename(5,(0.03,np.pi),(0.01,0.01,'Lorentz'),'v1',(-2.2,-0.9,0.5),data_dirname,(0,100)))
#Minimization
N = 4
pars_spread = (0.01,0.04,'Gauss')
phi = np.pi
factor_k = 5
pts_path = pic.shape[1]//factor_k
path = fs.path_BZ_KGK(fs.a_mono[0],pts_path,K_lim)
args_minimization = (N,pars_spread,phi,Hopt,bounds_pic,path,pic,True)
A_M, V = np.load(fs.compute_bopt_filename(data_dirname,args_minimization))
args_pic = (N,pic.shape[:2],path,fs.get_RLV(A_M))
pars_V = (V,phi)
print(A_M,V)
#
picture = fs.compute_image(pars_V,Hopt,pars_spread,bounds_pic,*args_pic)
fs.plot_image((picture,pic,np.absolute(picture-pic[:,:,0])),bounds_pic)
exit()
#
for V in [bV,]:#[0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04]:
    for A_M in [bA_M,]:#[30,40,50,60,70,80]:
        for phi in [np.pi,]:#[np.pi/4,np.pi]:
            print(A_M,V,phi)
            pars_V = (V,phi)
            filename = 'temp_gauss/'+"bbbbb_"+"{:.2f}".format(A_M)+'_'+"{:.4f}".format(pars_V[0])+'_'+"{:.4f}".format(pars_V[1])+'_'+"{:.4f}".format(pars_spread[0])+'_'+"{:.4f}".format(pars_spread[1])+'_'+pars_spread[2]+'.png'
            if 0:#Path(filename).is_file():
                continue
            args = other_args = (N,pic.shape[:2],fs.path_BZ_KGK(fs.a_mono[0],pts_path,K_lim),fs.get_RLV(A_M))
            picture = fs.compute_image(pars_V,Hopt,pars_spread,bounds_pic,*args)
            #
            fs.plot_final(pic,picture,bounds_pic,filename,pars_V,A_M)


