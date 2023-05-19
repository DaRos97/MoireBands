import numpy as np
import functions as fs
import parameters as PARS

####not in cluster
import tqdm
def banana_lorentz(args):
    N,upper_layer,lower_layer,pts_ps,Path,sbv,factor_gridy,E_,K_,larger_E,dirname = args
    #
    E_cut = 0.4
    #
    data_name = dirname + "en_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    weights_name = dirname + "arpes_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    res = np.load(data_name)
    weight = np.load(weights_name)
    a_mono = [PARS.dic_params_a_mono[upper_layer],PARS.dic_params_a_mono[lower_layer]]
    path,K_points = fs.pathBZ(Path,a_mono[0],pts_ps)
    #
    dic_sym = {'G':r'$\Gamma$', 'K':r'$K$', 'Q':r'$K/2$', 'q':r'$-K/2$', 'M':r'$M$', 'm':r'$-M$', 'N':r'$M/2$', 'n':r'$-M/2$', 'C':r'$K^\prime$', 'P':r'$K^\prime/2$', 'p':r'$-K^\prime/2$'}
    bnds = len(res[0,0,:])
    #parameters of Lorentzian
    lp = len(path);     gridx = lp;    #grid in momentum fixed by points evaluated previously 
    gridy = lp*factor_gridy
    K2 = K_**2
    E2 = E_**2
    min_e = np.amin(np.ravel(res))
    max_e = np.amax(np.ravel(res))
    MIN_E = min_e - larger_E
    MAX_E = max_e + larger_E
    delta = MAX_E - MIN_E
    step = delta/gridy
    #K-axis
    #Ki, Km, Kf = K_points
    Ki = K_points[0]
    Kf = K_points[-1]
    K_list = np.linspace(-np.linalg.norm(Ki),np.linalg.norm(Kf),lp)
    E_list = np.linspace(MIN_E,MAX_E,gridy)
    #Compute values of lorentzian spread of weights
    lor_name = dirname + "FC_"+lower_layer+"-"+upper_layer+"_"+str(N)+'_'+Path+'_'+str(pts_ps)
    par_name = '_Full_('+str(gridy)+'_'+str(larger_E).replace('.',',')+'_'+str(K_).replace('.',',')+'_'+str(E_).replace('.',',')+')'+'_'+str(sbv[0])+'_'+str(sbv[1])+".npy"
    lor_name += par_name
    try:
        lor = np.load(lor_name)
        print("\nLorentzian spread already computed")
    except:
        print("\nComputing Lorentzian spread ...")
        lor = np.zeros((lp,gridy))
        for i in tqdm.tqdm(range(lp)):
            for l in range(2):
                for j in range(bnds):
                    #if res[l,i,j] < MIN_E or res[l,i,j] > MAX_E:
                    #    continue
                    pars = (K2,E2,weight[l,i,j],K_list[i],res[l,i,j])
                    lor += fs.lorentzian_weight(K_list[:,None],E_list[None,:],*pars)
        np.save(lor_name,lor)
