import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import sys

#sample = '3'
#sample = '11'
sample = sys.argv[1]
print("Computing side band distance of sample "+sample)

dirname = "Figs/"
dirname_data = "Data/"
#
n_aM = 121 #number of aM points to evaluate
aM = np.linspace(fs.moire_length(0),20,n_aM)
#Mass
fit_pars_fn = dirname_data + "S"+sample+"_fit_parameters.npy"
popt_fit = np.load(fit_pars_fn)
M = popt_fit[0]
#Cut exp results
exp_cut_fn = dirname_data + "S"+sample+"_cut_data.npy"
exp_data = np.load(exp_cut_fn)

V = 0.01
phi = np.pi*0

if 0:   #Plot KGK and MGM bands
    aM = 45
    g = 2*np.pi/aM
    if 0:    #KGK
        eta = 0
        listk = np.linspace(-4*g,4*g,300)
        en = np.zeros((len(listk),7))
        we = np.zeros((len(listk),7))
        for i in range(len(listk)):
            H = fs.Ham(np.array([listk[i],0]),g,M,V,phi,eta)
            en[i],ev = np.linalg.eigh(H)
            for b in range(7):
                we[i,b] = np.abs(ev[0,b])**2
        E = np.max(en[:,-1])-exp_data[0,0]
        fig,ax = plt.subplots()
        for i in range(7):
            ax.plot(listk,en[:,i],c='k',lw=0.3)
            ax.scatter(listk,en[:,i],s=we[:,i]*30,c='b',lw=0)
        ax.plot([-4*g,4*g],[E,E],c='r')
        ax.set_ylim(2*E,5*np.max(en[:,-1]))
        plt.show()
    else:   #MGM
        N = 10
        for n in range(N):
            eta = np.pi/6/(N-1)*n
            listk = np.linspace(-5*g,5*g,500)
            en = np.zeros((len(listk),7))
            we = np.zeros((len(listk),7))
            for i in range(len(listk)):
                H = fs.Ham(np.array([listk[i],0]),g,M,V,phi,eta)
                en[i],ev = np.linalg.eigh(H)
                we[i] = np.absolute(ev[0])**2
            E = np.max(en[:,-1])-exp_data[0,0]
            fig,ax = plt.subplots()
            fig.set_size_inches(15,15)
            for i in range(7):
                ax.plot(listk,en[:,i],c='k',lw=0.3)
                ax.scatter(listk,en[:,i],s=we[:,i]*30,c='b',lw=0)
            ax.plot([-4*g,4*g],[E,E],c='r')
            ax.set_ylim(2*E,5*np.max(en[:,-1]))
            plt.show()
    exit()


##############################################################
fig,ax = plt.subplots()
fig.set_size_inches(10,8)
ax_up = ax.twiny()
s_ = 20

if 1:####################################################################### External band
    energy = exp_data[0,0]      #energy of the cut taken from the VBM
    exp_dist_ext = exp_data[0,1]    #momentum distance between maion and side band

    #d_e_K_analytic = np.zeros(n_aM)
    #d_e_M_analytic = np.zeros(n_aM)
    d_e_K_numeric = np.zeros(n_aM)
    d_e_M_numeric = np.zeros(n_aM)

    d_e_rot = np.zeros((n_aM,2)) #distance of first and second external bands

    for i in range(n_aM):
        G = 2*np.pi/aM[i]
        #d_e_K_analytic[i] = fs.dist_ext_KGK_an(G,M,energy,V)
        #d_e_M_analytic[i] = fs.dist_ext_MGM_an(G,M,energy,V)
        #d_e_K_numeric[i] = fs.dist_ext_KGK_num(G,M,energy,V,phi)
        #d_e_M_numeric[i] = fs.dist_ext_MGM_num(G,M,energy,V,phi)
        d_e_rot[i] = fs.dist_ext_rot(G,M,energy,V,phi)
    #
    ind_Am1 = np.argmin(abs(d_e_rot[:,0]-exp_dist_ext))
    ind_Am2 = np.argmin(abs(d_e_rot[:,1]-exp_dist_ext))
    #ax.plot(aM,d_e_K_analytic,'g',label=r'$K\Gamma K$')
    #ax.plot(aM,d_e_M_analytic,'limegreen',label=r'$M\Gamma M$')
    #ax.plot(aM,d_e_K_numeric,'g',ls='dashed',label=r'$K\Gamma K$ num')
    #ax.plot(aM,d_e_M_numeric,'limegreen',ls='dashed',label=r'$M\Gamma M$ num')
    #
    ax.plot(aM,d_e_rot[:,0],'r',ls='dashed',label='first sideband: '+"{:.1f}".format(fs.twist_angle(aM[ind_Am1])/np.pi*180)+"°")
    ax.plot(aM,d_e_rot[:,1],'m',ls='dashed',label='second sideband: '+"{:.1f}".format(fs.twist_angle(aM[ind_Am2])/np.pi*180)+"°")
    #
    ax.plot([aM[0],aM[-1]],[exp_dist_ext,exp_dist_ext],'aqua',label='exp S'+sample)
    ax.axvline(aM[ind_Am1],c='r')
    ax.axvline(aM[ind_Am2],c='m')
    #
    ax.set_ylim(0,2*exp_dist_ext)
    ax.set_ylabel(r'$\lambda$ ($\mathring{A}^{-1}$)',size=s_)
    ax.legend(fontsize=s_-10,loc='upper left')

if 0:####################################################################### Up band
    ax_r = ax.twinx()
    momentum = abs(exp_data[1,0])
    exp_dist_up = exp_data[1,1]

    d_u_K_analytic = np.zeros(n_aM)
    d_u_M_analytic = np.zeros(n_aM)
    d_u_K_numeric = np.zeros(n_aM)
    d_u_M_numeric = np.zeros(n_aM)

    for i in range(n_aM):
        G = 2*np.pi/aM[i]
        d_u_K_analytic[i] = fs.dist_up_KGK_an(G,M,momentum,V,phi)
        d_u_M_analytic[i] = fs.dist_up_MGM_an(G,M,momentum,V,phi)
        d_u_K_numeric[i] = fs.dist_up_KGK_num(G,M,momentum,V,phi)
        d_u_M_numeric[i] = fs.dist_up_MGM_num(G,M,momentum,V,phi)
    #
    ax_r.plot(aM,d_u_K_analytic,'firebrick',label=r'$K\Gamma K$')
    ax_r.plot(aM,d_u_M_analytic,'r',label=r'$M\Gamma M$')
    ax_r.plot(aM,d_u_K_numeric,'firebrick',ls='dashed',label=r'$K\Gamma K$ num')
    ax_r.plot(aM,d_u_M_numeric,'r',ls='dashed',label=r'$M\Gamma M$ num')
    #
    ax_r.set_ylim(0,2*exp_dist_up)
    ax_r.set_xlabel(r'$a_M$ ($\mathring{A}$)',size=s_)
    ax_r.set_ylabel(r'$\gamma$ ($eV$)',size=s_)
    ax_r.legend(fontsize=s_-10,loc='upper right')

#
xtv = []
xtl = []
xutl = []
nt = 7
up_theta = False
for i in range(nt+1):
    ind = int(n_aM/nt*i) if not i == nt else -1
    xtv.append(aM[ind])
    xtl.append("{:.1f}".format(aM[ind]))
    if up_theta:
        xutl.append("{:.1f}".format(fs.twist_angle(aM[ind])/np.pi*180))
    else:
        xutl.append("{:.1f}".format(fs.miniBZ_rotation(fs.twist_angle(aM[ind]))/np.pi*180)+"°")
        xtl[-1] += " ("+"{:.1f}".format(fs.twist_angle(aM[ind])/np.pi*180)+"°)"

ax.set_xticks(xtv,xtl)
ax.set_xlim(aM[0],aM[-1])

ax_up.set_xticks(xtv,xutl)
if up_theta:
    ax.set_xlabel(r'$a_M$ ($\mathring{A}$)',size=s_)
    ax_up.set_xlabel(r"twist angle $\theta$",size=s_)
else:
    ax.set_xlabel(r'$a_M$ ($\mathring{A}$), $\theta$ (deg)',size=s_)
    ax_up.set_xlabel(r"miniBZ rotation $\eta$",size=s_)

ax_up.set_xlim(aM[0],aM[-1])

ax.set_title("S"+sample+", V="+"{:.3f}".format(V)+r", $\phi=$"+"{:.2f}".format(phi))

fig.savefig(dirname+"S"+sample+"_V="+"{:.3f}".format(V)+"_phi="+"{:.2f}".format(phi)+".png")
plt.show()
