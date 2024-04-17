import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import parameters as ps
import os,sys

machine = 'loc'
final_fit_dn = 'temp/' #or ''

for ind in range(0,20):
    fixed_SO,range_par = fs.get_parameters_plot(ind)

    for TMD in fs.TMDs:
        print("Computing TMD: ",TMD,", in fixed SO: ",str(fixed_SO)," and range: ",range_par)
        exp_data = fs.get_exp_data(TMD,machine)
        pars = [0,]
        for file in os.listdir(fs.get_res_dn(machine)+final_fit_dn):
            terms = file.split('_')
            if terms[1] == TMD and terms[2]=="{:.2f}".format(range_par).replace('.',',') and str(fixed_SO)==terms[3]:
                pars = np.load(fs.get_res_dn(machine)+final_fit_dn+file)
                chi2 = terms[-1][:-4]
                break
        if len(pars)==1:
            print("Parameters not found for TMD: ",TMD,", range_par: ",range_par," and fixed SO: ",str(fixed_SO))
            continue
        if fixed_SO:
            SO_values = ps.initial_pt[TMD][-2:]
            full_pars = list(pars)
            for i in range(2):
                full_pars.append(SO_values[i])
        else:
            full_pars = pars
        tb_en = fs.energy(full_pars,exp_data,TMD)
        #
        pars_DFT = ps.initial_pt[TMD]
        DFT_en = fs.energy(pars_DFT,exp_data,TMD)
        title = "TMD: "+TMD+", range_par: "+"{:.2f}".format(range_par)+" on fixed SO: "+str(fixed_SO)+'. Chi2='+chi2
        #
        fig = fs.plot_together(exp_data,DFT_en,tb_en,title)
            
        fig.savefig(fs.get_fig_fn(TMD,range_par,fixed_SO,machine))
        plt.close(fig)

