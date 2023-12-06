import numpy as np
from scipy.optimize import minimize
from PIL import Image
import functions as fs
import os

cluster = False if os.getcwd()[6:11]=='dario' else True
home_dirname = "/home/users/r/rossid/3_new_simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/3_new_simple_model/"
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
N = 5
pars_spread = (0.01,0,'Gauss')       #energy is in minimization now
phi = np.pi
factor_k = 5
pts_path = pic.shape[1]//factor_k
path = fs.path_BZ_KGK(fs.a_mono[0],pts_path,K_lim)
args_minimization = (N,pars_spread,phi,Hopt,bounds_pic,path,pic,True)
result = minimize(
        fs.difference_bopt,
        x0 = (60,0.02,0.03),
        bounds = [(30,90),(0.001,0.05),(0.01,0.1)],
        args = args_minimization,
        method = 'Nelder-Mead',
        options = {
            'disp':not cluster,
            }
        )
print(result)

#Save result
bopt_filename = fs.compute_bopt_filename(data_dirname,args_minimization)
np.save(bopt_filename,result.x)
bopt_figname = fs.compute_bopt_figname(data_dirname,args_minimization)
args_minimization = (N,pars_spread,phi,Hopt,bounds_pic,path,pic,False)
picture = fs.difference_bopt(result.x,*args_minimization)
np.save(bopt_figname,picture)

if 0:
    fs.plot_image((picture,pic,np.absolute(picture-pic[:,:,0])),bounds_pic)
