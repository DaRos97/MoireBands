import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys,os
sys.path.insert(1,'/home/dario/Desktop/git/MoireBands/Code')
import CORE_functions as cfs
sys.path.insert(1,'/home/dario/Desktop/git/MoireBands/Code/1_monolayer')
import functions_monolayer as fsm
machine = cfs.get_machine(os.getcwd())          #Machine on which the computation is happening

TMD = 'WSe2'

fig_fn = 'Figures/superimposed_image/starting_point.png'
#Start importing image and get pixels of the 2 monolayers
pic_0 = np.array(np.asarray(Image.open(fig_fn)))

WS2_e1 = 129        #-1   eV
WS2_e2 = 390        #-2.5 eV
WS2_k1 = 112
WS2_k2 = 551

WSe2_e1 = 62        #0  eV
WSe2_e2 = 408       #-2 eV
WSe2_k1 = 718
WSe2_k2 = 1157

en_dic = {'WS2': (-1,-2.5), 'WSe2': (0,-2)}

pic_dic = {'WS2': pic_0[WS2_e1:WS2_e2,WS2_k1:WS2_k2], 'WSe2': pic_0[WSe2_e1:WSe2_e2,WSe2_k1:WSe2_k2]}

pic = pic_dic[TMD]

#Import parameters
pars_fn = '/home/dario/Desktop/git/MoireBands/Code/1_monolayer/Figures/result_'+TMD+'.npy'
fit_pars = np.load(pars_fn)
DFT_pars = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC

HSO_fit = cfs.find_HSO(fit_pars[-2:])
HSO_DFT = cfs.find_HSO(DFT_pars[-2:])


ind_reduced = 13
exp_data = fsm.get_exp_data(TMD,machine)
symm_data = fsm.get_symm_data(exp_data)
reduced_data = fsm.get_reduced_data(symm_data,ind_reduced)

for i in range(len(fit_pars)):
    print('index: ',i,' -> ',cfs.list_formatted_names_all[i],': ',"{:.7f}".format(fit_pars[i]))

#en_p_e 
fit_pars[6] += 0.2
#en_p_e 
fit_pars[2] += 0.2

en_fit = cfs.energy(fit_pars,HSO_fit,reduced_data,TMD)
#en_DFT = cfs.energy(DFT_pars,HSO_fit,reduced_data,TMD)


pe,pk = pic.shape[:2]
E_max, E_min = en_dic[TMD]

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot()
ax.imshow(pic)
for b in range(2):
    targ = np.argwhere(np.isfinite(reduced_data[b][:,1]))    #select only non-nan values
    k_list = reduced_data[b][targ,0]/np.max(reduced_data[b][targ,0])*pk
    ax.plot(k_list,(E_max-en_fit[b,targ])/(E_max-E_min)*pe,
            color='skyblue',
            marker='s',
            ls='-',
            label='Fit' if b == 0 else '',
            zorder=3,
            markersize=10,
            mew=1,
            mec='k',
            mfc='deepskyblue'
           )
    continue
    ax.plot(k_list,(E_max-en_DFT[b,targ])/(E_max-E_min)*pe,
            color='orange',
            marker='^',
            ls='-',
            label='DFT' if b == 0 else '',
            zorder=2,
            markersize=10,
            mew=1,
            mec='k',
            mfc='darkorange'
           )
ax.set_xlim(0,pk)

s_ = 20
ax.legend(fontsize=s_)
ks = [0,int(pk/3*2),pk]
ax.set_xticks(ks,[r"$\Gamma$",r"$K$",r"$M$"],size=s_)
for i in range(3):
    ax.axvline(ks[i],color='k',lw=0.5)
ax.set_ylabel('energy (eV)',size=s_)
label_y = []
ticks_y = np.linspace(0,pe,5)
for i in range(len(ticks_y)):
    label_y.append("{:.1f}".format(E_max+i*(E_min-E_max)/4))
ax.set_yticks(ticks_y,label_y,size=s_)

plt.tight_layout()
i#fig.savefig(TMD+'_superimposed.png')
plt.show()






















