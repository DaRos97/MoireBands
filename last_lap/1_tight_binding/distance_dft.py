import numpy as np
import parameters as ps
import functions as fs
import sys,os

from contextlib import redirect_stdout

final_fit_dn = 'temp/' #or ''

for ind in range(40):
    fixed_SO,range_par = fs.get_parameters_plot(ind)

    list_names = ps.list_names_all
    for TMD in fs.TMDs:
        tabname = fs.get_home_dn('loc') + 'results/Tables/DFT_vs_TB_'+TMD+'_'+str(fixed_SO)+'_'+"{:.2f}".format(range_par).replace('.',',')+'.txt'
        with open(tabname, 'w') as f:
            with redirect_stdout(f):
                for file in os.listdir(fs.get_home_dn('loc')+'results/'+final_fit_dn):
                    if file[5:5+len(TMD)] == TMD and file[6+len(TMD):10+len(TMD)]=="{:.2f}".format(range_par).replace('.',',') and file[11+len(TMD):11+len(TMD)+len(cuts_fn)]==cuts_fn:
                        pars_computed = np.load(fs.get_home_dn('loc')+'results/'+final_fit_dn+file)
                        chi2 = file[-10:-4]
                        break
                print("TMD: "+TMD+", range_par: "+"{:.2f}".format(range_par)+" on fixed SO: "+str(fixed_SO)+'. Chi2='+chi2,'\n')
                pars_dft = ps.initial_pt[TMD]
                if fixed_SO:
                    SO_values = ps.initial_pt[TMD][-2:]
                    full_pars = list(pars)
                    for i in range(2):
                        full_pars.append(SO_values[i])
                else:
                    full_pars = pars
                for i in range(len(pars_dft)):
                    if i != len(pars_dft)-1:
                        percentage = np.abs((full_pars[i]-pars_dft[i])/pars_dft[i]*100)
                        l = 15 - len(list_names[i])
                        print(list_names[i],':',' '*l,"{:.3f}".format(percentage),'%\t',"{:.5f}".format(pars_dft[i]),'\t->\t',full_pars[i])
                    else:
                        percentage = np.abs((full_pars[-1]-pars_dft[i])/pars_dft[i]*100)
                        l = 15 - len(list_names[i])
                        print(list_names[i],':',' '*l,"{:.3f}".format(percentage),'%\t',"{:.5f}".format(pars_dft[i]),'\t->\t',full_pars[-1])
                print('\n\n#############################\n\n')



















