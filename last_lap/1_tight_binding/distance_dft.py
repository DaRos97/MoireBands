import numpy as np
import parameters as ps
import functions as fs
import sys,os

from contextlib import redirect_stdout

final_fit_dn = 'temp/' #or ''

list_names = ps.list_names_all
for TMD in ['WSe2','WS2']:
    tabname = fs.get_home_dn('loc') + 'results/Table_DFT_vs_TB_'+TMD+'.txt'
    with open(tabname, 'w') as f:
        with redirect_stdout(f):
            print("TMD:\t",TMD,'\n')
            for file in os.listdir(fs.get_home_dn('loc')+'results/'+final_fit_dn):
                if file[5:5+len(TMD)] == TMD:
                    pars_computed = np.load(fs.get_home_dn('loc')+'results/'+final_fit_dn+file)
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



















