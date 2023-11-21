import numpy as np
import parameters as ps
import sys

from contextlib import redirect_stdout

list_names_all = ps.list_names_all
TMD = sys.argv[1]
consider_SO = True if sys.argv[2]=='True' else False
dirname = sys.argv[3]
txt_SO = "SO" if consider_SO else "noSO"

tabname = dirname + 'Table_DFT_vs_TB_'+TMD+'_'+txt_SO+'.txt'
with open(tabname, 'w') as f:
    with redirect_stdout(f):
        print("TMD:\t",TMD,'\n')
        filename = dirname + 'fit_pars_' + TMD + '_' + txt_SO + '.npy'
        pars_computed = np.load(filename)
        if txt_SO == 'SO':
            pars_dft = ps.initial_pt[TMD]
            list_names = list_names_all
        else:
            pars_dft = ps.initial_pt[TMD][:40]
            pars_dft.append(ps.initial_pt[TMD][-1])
            list_names = list_names_all[:40]
            list_names.append(list_names_all[-1])
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



















