import numpy as np
import functions as fs
import parameters as ps
import os,sys
import matplotlib.pyplot as plt

machine = 'loc'
final_dn = 'temp_20_10/'

type_bound = 'large'

TMD = fs.TMDs[0]
exp_data = fs.get_exp_data(TMD,machine)
symm_data = fs.get_symm_data(exp_data)
DFT_values = np.array(ps.initial_pt[TMD])  #DFT values
arr_chi21 = []

#
best_ind = 0
best_par_distance = 1e5
best_pars = np.zeros(DFT_values.shape)
best_chi2 = 1e5
for file in os.listdir(fs.get_res_dn(machine)+final_dn):
    terms = file.split('_')
    if terms[1] == TMD and type_bound==terms[3]:
        ind = int(terms[2])
        try:
            pars = np.load(fs.get_res_dn(machine)+final_dn+file)
        except:
            print("error in ind: ",ind)
            arr_chi21.append(np.array([ind,np.nan,0]))
            continue
        chi2 = float(terms[-1][:-4])
    #
        par_distance = fs.compute_parameter_distance(pars,DFT_values)
        #here we decide if we consider the solution with best overall chi2 or par_distance
        if par_distance<best_par_distance:
#        if chi2<best_chi2:
            best_par_distance = par_distance
            best_ind = ind
            best_pars = pars
            best_chi2 = chi2
        arr_chi21.append(np.array([ind,par_distance,chi2]))

arr_chi21 = np.array(arr_chi21)
if 1:   #plot par_distances
    plt.figure()
    plt.scatter(arr_chi21[:,0],arr_chi21[:,1],s=arr_chi21[:,2]*50,lw=0)
    plt.title("Distribution of result distances from DFT")
    plt.show()
    plt.figure()
    plt.scatter(arr_chi21[:,0],arr_chi21[:,2],s=arr_chi21[:,1]*10,lw=0)
    plt.title("Distribution of result distances from DFT")
    plt.show()

if not best_ind==0:
    print("Best sol found has ind=",best_ind," and chi2_1=",best_par_distance)
    HSO = fs.find_HSO(best_pars[-2:])
    tb_en = fs.energy(pars,fs.find_HSO(best_pars[-2:]),symm_data,TMD)
    DFT_en = fs.energy(DFT_values,fs.find_HSO(DFT_values[-2:]),symm_data,TMD)
    #
    plt.figure(figsize=(40,20))
    k_lim = exp_data[0][0][-1,0]
    ikl = exp_data[0][0].shape[0]//2
    title = "TMD: "+TMD
    s_ = 20
    for b in range(2):
        #exp
        plt.scatter(symm_data[b][:ikl,0],symm_data[b][:ikl,1],color='b',marker='*',label='experiment' if b == 0 else '')
        plt.scatter(-(symm_data[b][ikl:,0]-k_lim)+k_lim,symm_data[b][ikl:,1],color='b',marker='*')
        #DFT
        plt.scatter(symm_data[b][:ikl,0],DFT_en[b][:ikl],color='g',marker='^',s=1,label='DFT' if b == 0 else '')
        plt.scatter(-(symm_data[b][ikl:,0]-k_lim)+k_lim,DFT_en[b][ikl:],color='g',marker='^',s=1)
        #solution
        plt.scatter(symm_data[b][:ikl,0],tb_en[b][:ikl],color='r',marker='o',s=3,label='solution' if b == 0 else '')
        plt.scatter(-(symm_data[b][ikl:,0]-k_lim)+k_lim,tb_en[b][ikl:],color='r',marker='o',s=3)
    plt.legend(fontsize=s_,markerscale=2)
    plt.xticks([symm_data[b][0,0],symm_data[b][ikl,0],-(symm_data[b][-1,0]-k_lim)+k_lim],['$\Gamma$','$K$','$M$'],size=s_)
    plt.axvline(symm_data[b][0,0],color='k',alpha = 0.2)
    plt.axvline(symm_data[b][ikl,0],color='k',alpha = 0.2)
    plt.axvline(-(symm_data[b][-1,0]-k_lim)+k_lim,color='k',alpha = 0.2)
    plt.ylabel("E(eV)",size=s_)
    plt.suptitle(title,size=s_+10)
    plt.show()
    #Save best result
    if not input("Save result? [Y/n]")=='n':
        np.save(fs.get_res_dn(machine)+'result_'+TMD+'.npy',pars)
        os.system('python3 orbital_content.py')
else:
    print("No best solution found")

















