import numpy as np
import parameters as ps
import functions as fs
import sys,os

from contextlib import redirect_stdout

final_fit_dn = 'temp/' #or ''

for ind in range(20):
    considered_cuts,range_par = fs.get_parameters_plot(ind)
    cuts_fn = ''
    for i in range(len(considered_cuts)):
        cuts_fn += considered_cuts[i]
        if i != len(considered_cuts)-1:
            cuts_fn += '_'

    list_names = ps.list_names_all
    for TMD in ['WSe2','WS2']:
        tabname = fs.get_home_dn('loc') + 'results/Tables/DFT_vs_TB_'+TMD+'_'+cuts_fn+'_'+"{:.2f}".format(range_par).replace('.',',')+'.txt'
        with open(tabname, 'w') as f:
            with redirect_stdout(f):
                for file in os.listdir(fs.get_home_dn('loc')+'results/'+final_fit_dn):
                    if file[5:5+len(TMD)] == TMD and file[6+len(TMD):10+len(TMD)]=="{:.2f}".format(range_par).replace('.',',') and file[11+len(TMD):11+len(TMD)+len(cuts_fn)]==cuts_fn:
                        pars_computed = np.load(fs.get_home_dn('loc')+'results/'+final_fit_dn+file)
                        chi2 = file[-10:-4]
                print("TMD: "+TMD+", range_par: "+"{:.2f}".format(range_par)+" on cuts: "+cuts_fn+'. Chi2='+chi2,'\n')
                pars_dft = ps.initial_pt[TMD]
                for i in range(len(pars_dft)):
                    if i != len(pars_dft)-1:
                        percentage = np.abs((pars_computed[i]-pars_dft[i])/pars_dft[i]*100)
                        l = 15 - len(list_names[i])
                        print(list_names[i],':',' '*l,"{:.3f}".format(percentage),'%\t',"{:.5f}".format(pars_dft[i]),'\t->\t',pars_computed[i])
                    else:
                        percentage = np.abs((pars_computed[-1]-pars_dft[i])/pars_dft[i]*100)
                        l = 15 - len(list_names[i])
                        print(list_names[i],':',' '*l,"{:.3f}".format(percentage),'%\t',"{:.5f}".format(pars_dft[i]),'\t->\t',pars_computed[-1])
                print('\n\n#############################\n\n')



















