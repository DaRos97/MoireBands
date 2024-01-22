import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import parameters as ps
import os,sys

machine = 'loc'
final_fit_dn = 'temp/' #or ''

for ind in range(20):
    considered_cuts,range_par = fs.get_parameters_plot(ind)
    cuts_fn = ''
    for i in range(len(considered_cuts)):
        cuts_fn += considered_cuts[i]
        if i != len(considered_cuts)-1:
            cuts_fn += '_'
    print("Computing TMD: ",TMD,", in cuts: ",cuts_fn," and range: ",range_par)

    cuts_all = ['KGK','KMKp']

    for TMD in ['WSe2','WS2']:
        exp_data = fs.get_exp_data(TMD,cuts_all,machine)
        pars = [0]
        for file in os.listdir(fs.get_home_dn(machine)+'results/'+final_fit_dn):
            if file[5:5+len(TMD)] == TMD and file[6+len(TMD):10+len(TMD)]=="{:.2f}".format(range_par).replace('.',',') and file[11+len(TMD):11+len(TMD)+len(cuts_fn)]==cuts_fn:
                pars = np.load(fs.get_home_dn(machine)+'results/'+final_fit_dn+file)
                chi2 = file[-10:-4]
        if len(pars)==1:
            print("Parameters not found for TMD: ",TMD,", range_par: ",range_par," and cuts: ",cuts_fn)
            continue
        tb_en = fs.energy(pars,exp_data,cuts_all,TMD)
        #
        pars_DFT = ps.initial_pt[TMD]
        DFT_en = fs.energy(pars_DFT,exp_data,cuts_all,TMD)
        title = "TMD: "+TMD+", range_par: "+"{:.2f}".format(range_par)+" on cuts: "+cuts_fn+'. Chi2='+chi2
        #
        fig = fs.plot_together(exp_data,DFT_en,tb_en,title)
        fig.savefig(fs.get_fig_fn(TMD,considered_cuts,range_par,machine))
        plt.close(fig)

