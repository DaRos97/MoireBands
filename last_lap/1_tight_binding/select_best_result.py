import sys,os
import numpy as np
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:43]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/last_lap'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/last_lap'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions1 as fs
import matplotlib.pyplot as plt
from pathlib import Path

plots = 1
plots_pts = 0

selection = 0 if len(sys.argv)==1 else int(sys.argv[1])  #0->chi2, 1->chi2_0, 2->chi2_1
ind_0 = 0 if len(sys.argv) in [1,2] else int(sys.argv[2])   #index of spec_args
ind_reduced = 7 #Take 1 every .. pts in the exp data -> faster
txt_b = ['chi2','chi2_0','chi2_1']

machine = 'loc'

TMD,P,rp,rl = fs.get_spec_args(ind_0)
if 0:
    TMD = 'WSe2'
    P = 4
    rp = 3
    rl = 0
spec_args = (TMD,P,rp,rl,ind_reduced)
SOC_pars = np.load(fs.get_SOC_fn(TMD,machine))
H_SO = cfs.find_HSO(SOC_pars[1:])

print("Computing parameters: ",spec_args)
print("Considering solution with best ",txt_b[selection])

exp_data = fs.get_exp_data(TMD,machine)
symm_data = fs.get_symm_data(exp_data)
reduced_data = fs.get_reduced_data(symm_data,ind_reduced)
args_chi2 = (reduced_data,H_SO,SOC_pars,machine,spec_args,-1)
DFT_values = np.array(cfs.initial_pt[TMD])  #DFT values
arr_sol = []

#
best_i0 = -1
best = [1e5,1e5,1e5]
for file in os.listdir(fs.get_temp_dn(machine,spec_args)):
    terms = file.split('_')
    ind = int(terms[1])
    try:
        pars = np.load(fs.get_temp_dn(machine,spec_args)+file)
    except:
        print("error in ind: ",ind)
        arr_sol.append(np.array([ind,np.nan,np.nan,np.nan]))
        continue
    chi2 = float(terms[-1][:-4])    #remove the .npy
    chi2_1 = fs.compute_parameter_distance(pars,TMD)
    chi2_0 = chi2-P*chi2_1
    arr_sol.append(np.array([ind,chi2,chi2_0,chi2_1]))
    #
    temp = [chi2, chi2_0, chi2_1]
    if temp[selection]<best[selection]:
        best_i0 = len(arr_sol)-1
        best[selection] = temp[selection]
        best_pars = pars
arr_sol = np.array(arr_sol)

if plots_pts:   #plot par_distances
    plt.figure()
    plt.scatter(arr_sol[:,0],arr_sol[:,3],lw=0)
    plt.scatter(arr_sol[best_i0,0],arr_sol[best_i0,3],lw=0,c='r')
    plt.title("Distribution of chi2_1 from DFT")
    plt.show()
    plt.figure()
    plt.scatter(arr_sol[:,0],arr_sol[:,2],lw=0)
    plt.scatter(arr_sol[best_i0,0],arr_sol[best_i0,2],lw=0,c='r')
    plt.title("Distribution of chi2_0")
    plt.show()

if not best_i0==-1:
    print("Best sol found has ind=",arr_sol[best_i0,0]," and (chi2, chi2_0, chi2_1) = (","{:.4f}".format(arr_sol[best_i0,1]),", ","{:.4f}".format(arr_sol[best_i0,2]),", ","{:.4f}".format(arr_sol[best_i0,3]),')')
    print("__________________________________________________________")
    print("__________________________________________________________")
    #Save best result
    best_fn = fs.get_res_fn(spec_args,machine)
    if Path(best_fn).is_file():
        os.system('rm '+best_fn)
    np.save(best_fn,best_pars)
    if 1:
        fs.get_orbital_content(spec_args,machine,best_fn)
        fs.get_table(spec_args,machine,best_fn)
    #
    if plots:
        full_pars = np.append(best_pars,SOC_pars)
        tb_en = cfs.energy(full_pars,H_SO,reduced_data,TMD)
        DFT_en = cfs.energy(DFT_values,cfs.find_HSO(DFT_values[-2:]),reduced_data,TMD)
        #
        plt.figure(figsize=(40,20))
        title = "Spec_arg_ind = "+str(ind_0)+" -> ("+fs.get_spec_args_txt(spec_args)+")"+", chi2="+"{:.3f}".format(arr_sol[best_i0,1])
        s_ = 20
        for b in range(2):
            #exp
            plt.plot(reduced_data[b][:,0],reduced_data[b][:,1],color='b',marker='*',label='experiment' if b == 0 else '')
#            plt.plot(-(reduced_data[b][ikl:,0]-k_lim)+k_lim,reduced_data[b][ikl:,1],color='b',marker='*')
            #DFT
            plt.plot(reduced_data[b][:,0],DFT_en[b],color='g',marker='^',label='DFT' if b == 0 else '')
#            plt.scatter(-(reduced_data[b][ikl:,0]-k_lim)+k_lim,DFT_en[b][ikl:],color='g',marker='^',s=1)
            #solution
            plt.plot(reduced_data[b][:,0],tb_en[b],color='r',marker='o',label='solution' if b == 0 else '')
#            plt.scatter(-(reduced_data[b][ikl:,0]-k_lim)+k_lim,tb_en[b][ikl:],color='r',marker='o',s=3)
        plt.legend(fontsize=s_,markerscale=2)
        ikl = exp_data[0][0].shape[0]//2//ind_reduced+1
#        plt.xticks([reduced_data[b][0,0],reduced_data[b][ikl,0],reduced_data[b][-1,0]],['$\Gamma$','$K$','$M$'],size=s_)
        plt.axvline(reduced_data[b][0,0],color='k',alpha = 0.2)
        plt.axvline(reduced_data[b][ikl,0],color='k',alpha = 0.2)
        plt.axvline(reduced_data[b][-1,0],color='k',alpha = 0.2)
        plt.ylabel("E(eV)",size=s_)
        plt.suptitle(title,size=s_+10)
        plt.savefig(fs.get_fig_fn(spec_args,machine))
        if 1:
            print(full_pars)
            plt.show()

else:
    print("No best solution found")

















