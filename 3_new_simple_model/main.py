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
E_min_fig = -2.2
E_max_fig = -0.9
K_lim = 0.5
bounds_pic = (E_min_fig,E_max_fig,K_lim)
removed_k = (0,100)     #removed k points to identify upper and lower band
pars_V = (0.03,np.pi)
N = 5
pars_spread = (0.01,0.01,'Lorentz')
try:    #Reference image
    pic = np.load(fs.compute_picture_filename(version,bounds_pic,fig_dirname))
except:
    pic = fs.cut_image(bounds_pic,version,fig_dirname,True)
if 0: #plot image
    fs.plot_image(pic,bounds_pic)
#
factor_k = 10       #compute 1 every factor_k k-points in cut
pts_path = pic.shape[1]//factor_k
#
try:    #See if solution was already computed
    Hopt = np.load(fs.compute_Hopt_filename(N,pars_V,pars_spread,version,bounds_pic,data_dirname,removed_k))
    args = (N,pic.shape[:2],fs.path_BZ_KGK(fs.a_mono[0],pts_path,K_lim),fs.get_RLV(fs.A_M))
    fs.plot_image(fs.compute_image(pars_V,Hopt,pars_spread,bounds_pic,*args),bounds_pic)
    exit()
except:
    print("Computing...")
#
#Extract fit parameters of experimental data
try:
    pts_ul = np.load(fs.compute_pts_filename('ul',version,bounds_pic,data_dirname,removed_k[0]))
    pts_ll = np.load(fs.compute_pts_filename('ll',version,bounds_pic,data_dirname,removed_k[1]))
except:
    pts_ul, pts_ll = fs.compute_pts(pic,bounds_pic,version,data_dirname,removed_k,True)

#Minimize abs difference between darkest points of computed image and experimental data
#
path = fs.path_BZ_KGK(fs.a_mono[0],pts_path,K_lim)
G_M = fs.get_RLV(fs.A_M)
other_pars = (N,path)
pts_layers = (pts_ul,pts_ll)
arguments = (bounds_pic,pic,pts_layers,pars_V,pars_spread,N,path,G_M,removed_k)

#initial_point = np.array([0.126, -3.52, 0.608, 0.135, 0.532, -1.2])    
initial_point = np.array([0.1001, -3.181, 0.619, 0.1525, 0.5403, -1.2157])
optimistic_bounds = ((0.1,0.15),(-4,-3),(0.3,0.9),(0,0.3),(0.3,0.75),(-1.3,-1))
result = minimize(
        fs.abs_diff,
        args = arguments,
        x0 = initial_point,
        bounds = optimistic_bounds,
        method = 'Nelder-Mead',
        options = {
            'disp': False if cluster else True,
            'adaptive' : True,
            'maxiter': 1e6,
            },
        )

print("total: ",result)
np.save(fs.compute_Hopt_filename(N,pars_V,pars_spread,version,bounds_pic,data_dirname,removed_k),result.x)











