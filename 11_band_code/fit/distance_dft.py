import numpy as np
import parameters as ps
import sys

from contextlib import redirect_stdout

dirname = '../../Data/11_bands/'
list_names = [
            'e1', 
            'e3',   
            'e4',   
            'e6',   
            'e7',   
            'e9',   
            'e10',  
            't1_11',
            't1_22',   
            't1_33',   
            't1_44',   
            't1_55',   
            't1_66',   
            't1_77',   
            't1_88',   
            't1_99',   
            't1_1010',   
            't1_1111',
            't1_35',   
            't1_68',   
            't1_911',   
            't1_12',
            't1_34',   
            't1_45',   
            't1_67',   
            't1_78',   
            't1_910',   
            't1_1011',   
            't5_41',   
            't5_32',   
            't5_52',   
            't5_96',   
            't5_116',   
            't5_107',   
            't5_98',   
            't5_118',   
            't6_96',   
            't6_116',   
            't6_98',   
            't6_118',          
            'L_W',
            'L_S',
            'offset'
            ]

M = sys.argv[1]
consider_SO = True if sys.argv[2]=='True' else False
txt_SO = "SO" if consider_SO else "noSO"

tabname = 'Table_DFT_vs_TB_'+M+'_'+txt_SO+'.txt'
with open(tabname, 'w') as f:
    with redirect_stdout(f):
        print("TMD:\t",M,'\n')
        filename = dirname + 'fit_pars_' + M + '_' + txt_SO + '.npy'
        pars_computed = np.load(filename)
        pars_dft = ps.initial_pt[M]
        for i in range(len(pars_dft)):
            percentage = np.abs((pars_computed[i]-pars_dft[i])/pars_dft[i]*100)
            l = 15 - len(list_names[i])
            print(list_names[i],':',' '*l,"{:.3f}".format(percentage),'%\t',"{:.5f}".format(pars_dft[i]),'\t->\t',pars_computed[i])
        print('\n\n#############################\n\n')

