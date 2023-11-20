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
try:
    popts_ul = np.load(fs.compute_popts_filename('ul',version,bounds_pic,data_dirname))
    popts_ll = np.load(fs.compute_popts_filename('ll',version,bounds_pic,data_dirname))
except:
    fs.compute_popts(pic,bounds_pic,version,data_dirname,True)

#Minimize abs difference between darkest points of computed image and experimental data
pars_V = (0.033,np.pi)
#
spread_pars = (0.04,0.02,'Lorentz')
arguments = (bounds_pic,pic,popts_ul,popts_ll,pars_V,spread_pars)

optimistic_bounds = ((),(),)
result = minimize(
        fs.abs_diff,
        args = arguments,
        x0 = np.array([]),
        bounds = optimistic_bounds,
        method = 'Nelder-Mead',
        options = {
            'disp': False if cluster else True,
            'adaptive' : True,
            'fatol': 1e-8,
            'xatol': 1e-8,
            'maxiter': 1e6,
            },
        )

