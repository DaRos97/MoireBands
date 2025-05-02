import numpy as np
import matplotlib.pyplot as plt
import functions as fs
from pathlib import Path
from tqdm import tqdm

dirname = "Figs/"
dirname_data = "Data/"
#
n_aM = 121 #number of aM points to evaluate
aM = np.linspace(fs.moire_length(0),30,n_aM)
M = {}
exp_data = {}
for sample in ['3','11']:
    #Mass
    fit_pars_fn = dirname_data + "S"+sample+"_fit_parameters.npy"
    popt_fit = np.load(fit_pars_fn)
    M[sample] = popt_fit[0]
    #Cut exp results
    exp_cut_fn = dirname_data + "S"+sample+"_cut_data.npy"
    exp_data[sample] = np.load(exp_cut_fn)

nV = 21
list_V = np.linspace(0,0.05,nV)
phis = [0,np.pi/6,np.pi/3]

rres = np.zeros((len(phis),2,nV,2))

for p in range(len(phis)):
    phi = phis[p]
    fn = 'Data/results_'+"{:.1f}".format(phi/np.pi*180)+'_'+str(nV)+'.npy'

    if not Path(fn).is_file():
        #quantities to evaluate
        res = np.zeros((2,nV,2))

        for v in tqdm(range(nV)):
            d_e_rot = np.zeros((n_aM,2,2)) #distance of first and second external bands, for the 2 samples
            for i in range(n_aM):
                G = 2*np.pi/aM[i]
                d_e_rot[i] = fs.dist_ext_rot_full(G,M,exp_data,list_V[v],phi)
            #
            res[0,v,0] = fs.twist_angle(aM[np.argmin(abs(d_e_rot[:,0,0]-exp_data['3'][0,1]))])/np.pi*180
            res[0,v,1] = fs.twist_angle(aM[np.argmin(abs(d_e_rot[:,1,0]-exp_data['11'][0,1]))])/np.pi*180
            res[1,v,0] = fs.twist_angle(aM[np.argmin(abs(d_e_rot[:,0,1]-exp_data['3'][0,1]))])/np.pi*180
            res[1,v,1] = fs.twist_angle(aM[np.argmin(abs(d_e_rot[:,1,1]-exp_data['11'][0,1]))])/np.pi*180
        np.save(fn,res)
    else:
        res = np.load(fn)
    rres[p] = res

if 1:   #remove two results
    rres[0,1,16,0] = np.nan
    rres[0,1,17,0] = np.nan

s_ = 20
fig,ax = plt.subplots()
fig.set_size_inches(10,8)
from matplotlib import colormaps

cmap = colormaps['viridis']
c_s3 = cmap(np.linspace(0,1,7))
cmap = colormaps['plasma']
c_s11 = cmap(np.linspace(0,1,7))
for i in range(len(phis)):
    ax.plot(list_V,rres[i,0,:,0],c=c_s3[-3+i],marker='*',ls='-',label="sb1 S3")
    ax.plot(list_V,rres[i,1,:,0],c=c_s3[-3+i],marker='o',ls='-',label="sb2 S3")

    ax.plot(list_V,rres[i,0,:,1],c=c_s11[i],marker='*',ls='-',label="sb1 S11")
    ax.plot(list_V,rres[i,1,:,1],c=c_s11[i],marker='o',ls='-',label="sb2 S11")

ax.set_xlabel("V (eV)",size=s_)
ax.set_ylabel(r"$\theta$ (deg°)",size=s_)
ax.xaxis.set_tick_params(labelsize=s_)
ax.yaxis.set_tick_params(labelsize=s_)
#ax.set_title("moire phase: "+"{:.1f}".format(phi/np.pi*180)+'°')

import matplotlib.lines as mlin
p_sb1 = mlin.Line2D([],[],color='k',marker='*',markersize=10,lw=0,label="side band 1")
p_sb2 = mlin.Line2D([],[],color='k',marker='o',markersize=10,lw=0,label="side band 2")
ax.legend(handles=[p_sb1,p_sb2])

import matplotlib.patches as mpat
x, y = 0, 4
height, width = 1,0.002
rect1 = mpat.Rectangle((x, y), width, height, linewidth=1, edgecolor='none', facecolor=c_s3[-1])
rect2 = mpat.Rectangle((x, y), width, height/3*2, linewidth=1, edgecolor='none', facecolor=c_s3[-2])
rect3 = mpat.Rectangle((x, y), width, height/3, linewidth=1, edgecolor='none', facecolor=c_s3[-3])
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
x, y = 0.0035, 4
height, width = 1,0.002
rect1 = mpat.Rectangle((x, y), width, height, linewidth=1, edgecolor='none', facecolor=c_s11[2])
rect2 = mpat.Rectangle((x, y), width, height/3*2, linewidth=1, edgecolor='none', facecolor=c_s11[1])
rect3 = mpat.Rectangle((x, y), width, height/3, linewidth=1, edgecolor='none', facecolor=c_s11[0])
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
#text
ax.text(0,5.1,'S3',size=18)
ax.text(0.0029,5.1,'S11',size=18)

ax.text(0.0065,4.1,r'$\phi=0$°',size=18)
ax.text(0.0065,4.433,r'$\phi=30$°',size=18)
ax.text(0.0065,4.766,r'$\phi=60$°',size=18)

plt.show()

