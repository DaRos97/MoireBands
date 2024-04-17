import numpy as np
import parameters as ps
import functions as fs
import sys,os

from contextlib import redirect_stdout

final_fit_dn = 'temp/' #or ''

for ind in range(20):
    fixed_SO,range_par = fs.get_parameters_plot(ind)

    list_names = ps.list_names_all
    for TMD in fs.TMDs:
        tabname = fs.get_home_dn('loc') + 'results/tables/DFT_vs_fit_'+TMD+'_'+str(fixed_SO)+'_'+"{:.2f}".format(range_par).replace('.',',')+'.txt'
        with open(tabname, 'w') as f:
            with redirect_stdout(f):
                for file in os.listdir(fs.get_home_dn('loc')+'results/'+final_fit_dn):
                    terms = file.split('_')
                    if terms[1] == TMD and terms[2]=="{:.2f}".format(range_par).replace('.',',') and str(fixed_SO)==terms[3]:
                        pars_computed = np.load(fs.get_home_dn('loc')+'results/'+final_fit_dn+file)
                        chi2 = file[-10:-4]
                        break
                print("TMD: "+TMD+", range_par: "+"{:.2f}".format(range_par)+" on fixed SO: "+str(fixed_SO)+'. Chi2='+chi2,'\n')
                pars_dft = ps.initial_pt[TMD]
                if fixed_SO:
                    SO_values = ps.initial_pt[TMD][-2:]
                    full_pars = list(pars_computed)
                    for i in range(2):
                        full_pars.append(SO_values[i])
                else:
                    full_pars = pars_computed
                for i in range(len(pars_dft)):
                    percentage = np.abs((full_pars[i]-pars_dft[i])/pars_dft[i]*100)
                    l = 15 - len(list_names[i])
                    print(list_names[i],':',' '*l,"{:.3f}".format(percentage),'%\t',"{:.5f}".format(pars_dft[i]),'\t->\t',full_pars[i])
                print('\n\n#############################\n\n')



















