"""
Here we extract the results of the fit for the different parameters choices.
We compare chi2's of different results.
In the image we can compare the total chi2, the parameter distance from DFT and band content.
All plots are as a function of P (x-axis) for different rp (color), and rl (shape).
In the script we select by hand a single result of which are plotted bands, parameters and band content (sparse).
In the end an input option allows to save the final parameters of the fit and the various images.
"""
import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions_monolayer as fsm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
machine = cfs.get_machine(os.getcwd())          #Machine on which the computation is happening

if len(sys.argv) not in [1,2]:
    print("Usage: py select_best_result.py arg1")
    print("arg1: WSe2(default) or WS2")
    exit()
else:
    TMD = 'WSe2' if len(sys.argv)!=2 else sys.argv[1]

pars_selected_dic = {       #Best foud parameters: P, rp, rl
    'WSe2':[0.1,0.2  ,0],
#    'WSe2':[0.2,0.5 ,0.3],
    'WS2': [0.1,0.2  ,0]
                    }

#Loop over spec_args
nSA = 144        #number of specs -> see fsm.get_spec_args() 
pars_selected = pars_selected_dic[TMD]
lP = []
lrp = []
lrl = []
solutions_list = {}
#SOC_pars = cfs.initial_pt[TMD][-2:]
for ind_spec_args in range(nSA):
    spec_args = fsm.get_spec_args(ind_spec_args)
    if not TMD == spec_args[0]:
        spec_args = fsm.get_spec_args(ind_spec_args+nSA)
    if spec_args[1] not in lP:
        lP.append(spec_args[1])
    if spec_args[2] not in lrp:
        lrp.append(spec_args[2])
    if spec_args[3] not in lrl:
        lrl.append(spec_args[3])
    name = TMD+'_'+"{:.7f}".format(spec_args[1])+'_'+"{:.7f}".format(spec_args[2])+'_'+"{:.7f}".format(spec_args[3])
    solutions_list[name] = []
    #Loop over random realizations
    for file in os.listdir(fsm.get_temp_dn(machine,spec_args)):
        full_pars = np.load(fsm.get_temp_dn(machine,spec_args)+file)
        chi2 = float(file.split('_')[-1][:-4])    #remove the .npy
        chi2_ParDis = fsm.compute_parameter_distance(full_pars[:-2],TMD)
        band_content = np.array(fsm.compute_band_content(full_pars,cfs.find_HSO(full_pars[-2:]),TMD))
        chi2_BandCont = 2-np.sum(np.absolute(band_content)**2)
        if spec_args[1]==pars_selected[0] and spec_args[2]==pars_selected[1] and spec_args[3]==pars_selected[2]:
            result_pars = np.copy(full_pars)
            result_chi2 = chi2
            result_chi2_ParDis = chi2_ParDis
            result_chi2_BandCont = chi2_BandCont
        solutions_list[name].append(np.array([chi2,chi2_ParDis,chi2_BandCont]))

#Figure: x->P, y-> chi2
fig = plt.figure(figsize=(20,10))
titles = ["chi2","par distance","band content"]
for i_chi in range(3):
    ax = fig.add_subplot(1,3,i_chi+1)
    colors_rp = ['aqua','g','b','m','y','darkorange']       #need to be of same length as lrp in fsm.get_spec_args
    markers_rl = ['*','^','o','+']       #need to be of same length as lrp in fsm.get_spec_args
    for P in lP:
        for n in solutions_list.keys():
            if "{:.7f}".format(P) == n.split('_')[1]:
                color_rp = colors_rp[lrp.index(float(n.split('_')[2]))]
                marker_rl = markers_rl[lrl.index(float(n.split('_')[3]))]
                for ind_rand in range(len(solutions_list[n])):  #number of random evaluations
                    data = solutions_list[n][ind_rand][i_chi] if i_chi!=2 else solutions_list[n][ind_rand][0]+solutions_list[n][ind_rand][1]/5
                    ax.scatter(P,data,color=color_rp,marker=marker_rl,s=200)
                    if P==pars_selected[0] and n.split('_')[2]=="{:.7f}".format(pars_selected[1]) and n.split('_')[3]=="{:.7f}".format(pars_selected[2]):
                        ax.scatter(P,data,color='red',marker='o',zorder=100,s=100)
    #
    ax.set_title(titles[i_chi],size=20)
    ax.set_xlabel("P")
    if i_chi == 2:
        labels_rp = ["{:.3f}".format(rp) for rp in lrp]
        leg_en = [Line2D([0],[0],marker='o',color='none',label=label,markerfacecolor=color) for label,color in zip(labels_rp,colors_rp)]
        #
        labels_rl = ["{:.3f}".format(rl) for rl in lrl]
        for i in range(len(lrl)):
            leg_en.append(Line2D([0],[0],marker=markers_rl[i],color='none',label=labels_rl[i],markerfacecolor='k'))
        ax.legend(handles = leg_en)

plt.show()

if not input("Proceed with visualization of result?[y/N]")=='y':
    exit()


ind_reduced = 13
spec_args = [TMD,pars_selected[0],pars_selected[1],pars_selected[2],ind_reduced,10,20]
exp_data = fsm.get_exp_data(TMD,machine)
symm_data = fsm.get_symm_data(exp_data)
reduced_data = fsm.get_reduced_data(symm_data,ind_reduced)
#
tb_en = cfs.energy(result_pars,cfs.find_HSO(result_pars[-2:]),reduced_data,TMD)
dft_en = cfs.energy(cfs.initial_pt[TMD],cfs.find_HSO(result_pars[-2:]),reduced_data,TMD)
#
title = 'Final pars '+TMD+': P='+"{:.3f}".format(pars_selected[0])+', rp='+"{:.3f}".format(pars_selected[1])+', rl='+"{:.3f}".format(pars_selected[2])
fsm.plot_bands(tb_en,reduced_data,dft_en,title=title,show=True,TMD=TMD)
fsm.plot_parameters(result_pars,spec_args,title=title,show=True)
fsm.plot_orbitals(result_pars,title=title,show=True,TMD=TMD)

#Save
if input("Save selected result?[y/N]")=='y':
    print("Saving final results and figures")
    fsm.plot_bands(tb_en,reduced_data,dft_en,title=title,figname='Figures/bands_'+TMD+'.png',show=False,TMD=TMD)
    fsm.plot_parameters(result_pars,spec_args,title=title,figname='Figures/parameters_'+TMD+'.png',show=False)
    fsm.plot_orbitals(result_pars,title=title,figname='Figures/orbitals_'+TMD+'.png',show=False,TMD=TMD)
    np.save(fsm.get_res_fn(TMD,machine),result_pars)














