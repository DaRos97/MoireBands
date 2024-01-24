import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import functions as fs

mat = 'WS2'
n = 'up'
v = '2'
cuts = ['KGK','KMKp']
a_mono = {'WSe2':3.32, 'WS2':3.18}       #monolayer lattice lengths in Angstrom

#Extract points of GM
pts_filename = '/home/dario/Desktop/git/MoireBands/4_CEM_K/inputs/'+mat+'_'+n+'_v'+v+'.npy'
if not Path(pts_filename).is_file():
    GtoK = 0
    KtoM = 0
    res = []
    for cut in cuts:
        mono_filename = '/home/dario/Desktop/git/MoireBands/4_CEM_K/inputs/'+cut+'_'+mat+'_band_'+n+'_v'+v+'.txt'
        with open(mono_filename, 'r') as f:
            lines = f.readlines()
            N = len(lines)
            for i in range(N):
                temp = lines[i].split('\t')
                if i == N-1 and cut == 'KGK':
                    GtoK = float(temp[0])
                    E_last_G = float(temp[1])
                if i == 0 and cut == 'KMKp':
                    KtoM = abs(float(temp[0]))
                    E_first_M = float(temp[1])
                if temp[1]=='NAN\n':
                    continue
                res.append([float(temp[0]),float(temp[1])])
                if cut == 'KMKp':
                    res[-1][0] += GtoK+KtoM
                    res[-1][1] += E_last_G-E_first_M

    res = np.array(res)
    np.save(pts_filename,res)
else:
    res = np.load(pts_filename)

if 0:
    plt.figure()
    plt.plot(res[:,0],res[:,1],'r.')
    plt.title(mat,size=20)
    plt.xlabel(r"$K'\rightarrow\Gamma\rightarrow K\rightarrow M\rightarrow K'$",size=15)
    plt.ylabel("E (eV)",size=15)
    plt.show()
    exit()

#Extract fit pts in range from TVB
range_e = 0.15   #distance from TVB on which to do the fit, in eV
VBM = np.max(res[:,1])
range_k = 0.5
GtoK = 4*np.pi/3/a_mono[mat]
fit_pts = []
for i in range(len(res)):
    if res[i,0] > GtoK-range_k and res[i,0]<GtoK+range_k and res[i,1] > VBM-range_e:
        fit_pts.append([res[i,0],res[i,1]])
fit_pts = np.array(fit_pts)
if 0:
    plt.figure()
    plt.plot(fit_pts[:,0],fit_pts[:,1],'b.')
    plt.show()
    exit()

#Fit
#arg_VBM = np.argmax(fit_pts[:,1])
arg_VBM = np.argmin(np.absolute(fit_pts[:,0]-GtoK))

def par(k,m,mu):
    result = -(k-fit_pts[arg_VBM,0])**2/2/m + mu
    return result
def combined_par(k,m1,m2,mu):
    eg = par(k[:arg_VBM],m1,mu)
    em = par(k[arg_VBM:],m2,mu)
    return np.append(eg,em)
mu_0 = -0.5 if mat=='WSe2' else -1
popt,pcov = curve_fit(combined_par,fit_pts[:,0],fit_pts[:,1],p0=[0.03,0.1,mu_0])

print(popt)

if 1:
    plt.figure()
    plt.plot(fit_pts[:,0],fit_pts[:,1],'b.')
    x_line_g = np.linspace(fit_pts[0,0],fit_pts[arg_VBM,0],100)
    plt.plot(x_line_g,par(x_line_g,popt[0],popt[-1]),'r-')
    x_line_m = np.linspace(fit_pts[arg_VBM,0],fit_pts[-1,0],100)
    plt.plot(x_line_m,par(x_line_m,popt[1],popt[-1]),'g-')
    plt.title(mat+' around K with fit range '+"{:.4f}".format(range_e)+' eV')
    plt.xlabel(r'$A^{-1}$')
    plt.ylabel(r'$E$')
    import matplotlib.patches as mpatches
    G_patch = mpatches.Patch(color='r', label="$m_\Gamma=$"+"{:.4f}".format(popt[0]))
    M_patch = mpatches.Patch(color='g', label="$m_M=$"+"{:.4f}".format(popt[1]))
    plt.legend(handles=[G_patch,M_patch])
    fig = plt.gcf()
    plt.show()
    if input("Save? (y/N)")=='y':
        popt_filename = fs.Hopt_filename(mat,n,v)
        np.save(popt_filename,popt)
        fig_filename = 'inputs/fit_fig_'+mat+'_'+n+'_v'+v+'_range_'+"{:.4f}".format(range_e)+'.png'
        fig.savefig(fig_filename)

















