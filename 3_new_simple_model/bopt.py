import numpy as np
from scipy.optimize import minimize
from PIL import Image
import functions as fs
import os
from pathlib import Path

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
#Check
bopt_filename = fs.compute_bopt_filename(data_dirname,args_minimization)
bopt_figname = fs.compute_bopt_figname(data_dirname,args_minimization)
if Path(bopt_figname).is_file() and Path(bopt_filename).is_file():
    print("Already computed")
    if not cluster:
        bopt = np.load(bopt_filename)
        print(bopt)
        print(Hopt)
        picture = np.load(bopt_figname)
        fs.plot_image((picture,pic,np.absolute(picture-pic[:,:,0])),bounds_pic)
    exit()

result = minimize(
        fs.difference_bopt,
        x0 = (60,0.02,0.03,*Hopt),
        bounds = [(30,90),(0.001,0.05),(0.01,0.1),
            (0.05,0.15),(-3.3,-3.06),(0.5,0.72),(0.05,0.25),(0.44,0.64),(-1.31,-1.11)],
        args = args_minimization,
        method = 'Nelder-Mead',
        options = {
            'adaptive':True,
            'disp':not cluster,
            }
        )
print(result)

#Save result
np.save(bopt_filename,result.x)
args_minimization = (N,pars_spread,phi,Hopt,bounds_pic,path,pic,False)
picture = fs.difference_bopt(result.x,*args_minimization)
np.save(bopt_figname,picture)

if 0:
    fs.plot_image((picture,pic,np.absolute(picture-pic[:,:,0])),bounds_pic)
