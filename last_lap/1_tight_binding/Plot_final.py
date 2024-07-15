import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import parameters as ps
import os,sys

machine = 'loc'
final_fit_dn = 'temp/'

type_bound = 'fixed'
range_par = 0.1

for TMD in fs.TMDs:
    print("Plotting result of TMD: ",TMD," and range: ",range_par," ",type_bound)
    exp_data = fs.get_exp_data(TMD,machine)
    pars = [0,]
    for file in os.listdir(fs.get_res_dn(machine)+final_fit_dn):
        terms = file.split('_')
        if terms[1] == TMD and terms[3]=="{:.2f}".format(range_par).replace('.',',') and str(type_bound)==terms[2]:
            pars = np.load(fs.get_res_dn(machine)+final_fit_dn+file)
            chi2 = terms[-1][:-4]
            break
    if len(pars)==1:
        print("Parameters not found for TMD: ",TMD," and range: ",range_par," ",type_bound)
        continue
    #
    pars_DFT = ps.initial_pt[TMD]
    HSO = fs.find_HSO(pars_DFT[-2:])
    tb_en = fs.energy(pars,HSO,exp_data,TMD)
    args = (exp_data, TMD, machine, range_par, type_bound, HSO)
    comp_chi2 = fs.chi2(pars,*args)
    if 0:   #Just save energies
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
    DFT_en = fs.energy(pars_DFT[:-2],HSO,exp_data,TMD)
    title = "TMD: "+TMD+", range_par: "+"{:.2f}".format(range_par)+" eV, "+str(type_bound)+'. Chi2='+chi2        #
    print("computed chi^2: ",comp_chi2)
    fig = fs.plot_together(exp_data,DFT_en,tb_en,title)
    plt.show()
#        fig.savefig(fs.get_fig_fn(TMD,range_par,fixed_SO,machine))
#        plt.close(fig)

