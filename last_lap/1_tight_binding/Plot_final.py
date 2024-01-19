import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import parameters as ps
import os,sys

machine = 'loc'
final_fit_dn = 'temp/' #or ''

for TMD in ['WSe2','WS2']:
    exp_data = fs.get_exp_data(TMD,machine)
    for file in os.listdir(fs.get_home_dn(machine)+'results/'+final_fit_dn):
        if file[5:5+len(TMD)] == TMD:
            pars = np.load(fs.get_home_dn(machine)+'results/'+final_fit_dn+file)
    tb_en = fs.energy(pars,exp_data,TMD)
    #
    pars_DFT = ps.initial_pt[TMD]
    DFT_en = fs.energy(pars_DFT,exp_data,TMD)
    title = TMD+': chi2='+file[-10:-4]
    fs.plot_exp_tb(exp_data,DFT_en,tb_en,title)
