import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import parameters as ps
import os,sys

machine = 'loc'
final_fit_dn = 'temp/' #or ''

for ind in range(0,20):
    considered_cuts,range_par = fs.get_parameters_plot(ind)
    cuts_fn = fs.get_cuts_fn(considered_cuts)

    cuts_all = ['KGK','KMKp']

    for TMD in ['WSe2','WS2']:
        print("Computing TMD: ",TMD,", in cuts: ",cuts_fn," and range: ",range_par)
        exp_data = fs.get_exp_data(TMD,cuts_all,machine)
        if 0:   #Plot bands of exp next to each other to verify alignment
            plt.figure()
            for b in range(2):
                plt.scatter(exp_data[0][b][:,0],exp_data[0][b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
                plt.scatter(exp_data[1][b][:,0]+(exp_data[0][b][-1,0]-exp_data[1][b][0,0]),exp_data[1][b][:,1],color='m',marker='*',label='experiment' if b == 0 else '')
            plt.show()
            exit()
        pars = [0,]
        for file in os.listdir(fs.get_res_dn(machine)+final_fit_dn):
            terms = file.split('_')
            if terms[1] == TMD and terms[2]=="{:.2f}".format(range_par).replace('.',',') and len(terms)-4==len(considered_cuts):
                pars = np.load(fs.get_res_dn(machine)+final_fit_dn+file)
                chi2 = terms[-1][:-4]
                break
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
#        plt.show()
#        exit()
        if 1:#input("Save fig? (y/N)")=='y':
            fig.savefig(fs.get_fig_fn(TMD,considered_cuts,range_par,machine))
        plt.close(fig)

