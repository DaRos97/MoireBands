import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import parameters as ps
import os,sys

machine = 'loc'
final_fit_dn = 'temp/' #or ''

for ind in [1,3]:#range(0,20):
    fixed_SO,range_par = fs.get_parameters_plot(ind)

    for TMD in fs.TMDs:
        if ind == 1:
            TMD = 'WS2'
        elif ind == 3:
            TMD = 'WSe2'
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
        if 1:   #Just save energies
            for bb in range(2):
                fname = 'results/Data_GM/'+TMD+'_band'+str(bb+1)+'_11bandmodel.txt'
                plt.figure()
                plt.plot(np.array(exp_data[0][bb])[:,3],tb_en[0][0])
                plt.show()
                savefile = np.zeros((len(tb_en[0][bb]),3))
                savefile[:,0] = np.array(exp_data[0][bb])[:,2]
                savefile[:,1] = np.array(exp_data[0][bb])[:,3]
                savefile[:,2] = tb_en[0][bb]
                np.savetxt(fname,savefile,fmt='%.6e',delimiter='\t',
                        header='The three columns are: kx,ky,energy.'
                    )
        #
        pars_DFT = ps.initial_pt[TMD]
        DFT_en = fs.energy(pars_DFT,exp_data,TMD)
        title = "TMD: "+TMD+", range_par: "+"{:.2f}".format(range_par)+" on fixed SO: "+str(fixed_SO)+'. Chi2='+chi2
        #
        fig = fs.plot_together(exp_data,DFT_en,tb_en,title)
        fig.savefig(fs.get_fig_fn(TMD,range_par,fixed_SO,machine))
        plt.close(fig)

