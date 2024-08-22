import numpy as np
import functions as fs
import parameters as ps
import os,sys
import matplotlib.pyplot as plt

plots = 1
selection = 0 if len(sys.argv)==1 else int(sys.argv[1])  #0->chi2, 1->chi2_0, 2->chi2_1
txt_b = ['chi2','chi2_0','chi2_1']

machine = 'loc'
TMD = fs.TMDs[0]

P = 10
rp = 1.0
rl = 0.1
spec_args = (P,rp,rl)

print("Computing TMD: ",TMD," with parameters: ",spec_args)
print("Considering solution with best ",txt_b[selection])

TMD = fs.TMDs[0]
exp_data = fs.get_exp_data(TMD,machine)
symm_data = fs.get_symm_data(exp_data)
args_chi2 = (symm_data,TMD,machine,spec_args,0)
DFT_values = np.array(ps.initial_pt[TMD])  #DFT values
arr_sol = []

#
best_ind = 0
best_chi2_1 = 1e5       #parameters chi2
best_chi2_0 = 1e5       #energy chi2
best_pars = np.zeros(DFT_values.shape)
best_chi2 = 1e5
best = [best_chi2, best_chi2_0, best_chi2_1]
for file in os.listdir(fs.get_temp_dn(machine,spec_args)):
    terms = file.split('_')
    if terms[1] == TMD:
        ind = int(terms[2])
        try:
            pars = np.load(fs.get_temp_dn(machine,spec_args)+file)
        except:
            print("error in ind: ",ind)
            arr_sol.append(np.array([ind,np.nan,np.nan,np.nan]))
            continue
        chi2 = float(terms[-1][:-4])
        chi2_1 = fs.compute_parameter_distance(pars,DFT_values)
        chi2_0 = chi2-P*chi2_1
        temp = [chi2, chi2_0, chi2_1]
        #
        if temp[selection]<best[selection]:
            best_chi2_1 = chi2_1
            best_chi2_0 = chi2_0
            best_ind = ind
            best_pars = pars
            best_chi2 = chi2
        arr_sol.append(np.array([ind,chi2,chi2_0,chi2_1]))

arr_sol = np.array(arr_sol)
if 0 and plots:   #plot par_distances
    plt.figure()
    plt.scatter(arr_sol[:,0],arr_sol[:,3],s=arr_sol[:,2]*50,lw=0)
    plt.title("Distribution of chi2_1 from DFT")
    plt.show()
    plt.figure()
    plt.scatter(arr_sol[:,0],arr_sol[:,2],s=arr_sol[:,3]*10,lw=0)
    plt.title("Distribution of chi2_0")
    plt.show()

if not best_ind==0:
    print("Best sol found has ind=",best_ind," and (chi2_0, chi2_1) = (",best_chi2_0,", ",best_chi2_1,')')
    if plots:
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
    np.save(fs.get_res_fn(TMD,spec_args,machine),best_pars)
    fs.get_orbital_content(TMD,spec_args,machine)
    fs.get_table(TMD,spec_args,machine)
else:
    print("No best solution found")

















