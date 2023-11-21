import numpy as np
from scipy.optimize import minimize
from PIL import Image
import functions as fs


cluster = False
home_dirname = "/home/users/r/rossid/3_new_simple_model/" if cluster else "/home/dario/Desktop/git/MoireBands/3_new_simple_model/"
fig_dirname = home_dirname + "input_figures/"
data_dirname = home_dirname + "input_data/"

#Borders of image in terms of energy and momentum
version = "v1"
E_min_fig = -2.2
E_max_fig = -0.9
K_lim = 0.5
bounds_pic = (E_min_fig,E_max_fig,K_lim)
try:
    pic = np.load(fs.compute_picture_filename(version,bounds_pic,fig_dirname))
except:
    pic = fs.cut_image(bounds_pic,version,fig_dirname,True)
len_e, len_k, z = pic.shape
if 0: #plot image
    fs.plot_image(pic,bounds_pic)

#Extract fit parameters of experimental data
removed_k = (0,100)
try:
    pts_ul = np.load(fs.compute_pts_filename('ul',version,bounds_pic,data_dirname,removed_k[0]))
    pts_ll = np.load(fs.compute_pts_filename('ll',version,bounds_pic,data_dirname,removed_k[1]))
except:
    pts_ul, pts_ll = fs.compute_pts(pic,bounds_pic,version,data_dirname,removed_k,True)

#Minimize abs difference between darkest points of computed image and experimental data
pars_V = (0.03,np.pi)
N = 5
#
factor_k = 10
pts_path = len_k//factor_k
path = fs.path_BZ_KGK(fs.a_mono[0],pts_path,K_lim)
G_M = fs.get_RLV(fs.A_M)
other_pars = (N,path)
pars_spread = (0.01,0.01,'Lorentz')
pts_layers = (pts_ul,pts_ll)
arguments = (bounds_pic,pic,pts_layers,pars_V,pars_spread,N,path,G_M,removed_k)

initial_point = np.array([0.126, -3.52, 0.608, 0.135, 0.532, -1.2])
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
