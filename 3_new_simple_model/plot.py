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
E_min_fig = -2.2
E_max_fig = -0.9
K_lim = 0.5
bounds_pic = (E_min_fig,E_max_fig,K_lim)
removed_k = (0,100)
try:    #Reference image
    pic = np.load(fs.compute_picture_filename(version,bounds_pic,fig_dirname))
except:
    pic = fs.cut_image(bounds_pic,version,fig_dirname,True)
#Relevant parameters for importing minimization parameters of H
pars_V = (0.03,np.pi)
N = 5
pars_spread = (0.01,0.01,'Lorentz')
Hopt = np.load(fs.compute_Hopt_filename(N,pars_V,pars_spread,version,bounds_pic,data_dirname,removed_k))
#New parameters for exploring image
N = 5
factor_k = 5
pts_path = pic.shape[1]//factor_k
pars_spread = (0.01,0.04,'Gauss')
#
for V in [0.02,]:#[0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04]:
    for A_M in [70,]:#[30,40,50,60,70,80]:
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


