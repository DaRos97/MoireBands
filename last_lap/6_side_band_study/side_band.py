import numpy as np
import matplotlib.pyplot as plt
import functions as fs

nn = 100
aM = np.linspace(80,30,nn)
M = 0.37#0.13436
V = 0.00
phi = 0#np.pi

if 1:####################################################################### External band
    exp_dist_ext = 0.1114 #A^-1
    energy = -0.44      #eV, taken into account the offset

    d_e_K_analytic = np.zeros(nn)
    d_e_M_analytic = np.zeros(nn)
    d_e_K_numeric = np.zeros(nn)
    d_e_M_numeric = np.zeros(nn)

    for i in range(nn):
        G = 2*np.pi/aM[i]
        d_e_K_analytic[i] = fs.dist_ext_KGK_an(G,M,energy,V)
        d_e_M_analytic[i] = fs.dist_ext_MGM_an(G,M,energy,V)
        d_e_K_numeric[i] = fs.dist_ext_KGK_num(G,M,energy,V,phi)
        d_e_M_numeric[i] = fs.dist_ext_MGM_num(G,M,energy,V,phi)

if 1:####################################################################### Up band
    exp_dist_up = 0.0892 #eV
    momentum = 0.36       #A^-1

    d_u_K_analytic = np.zeros(nn)
    d_u_M_analytic = np.zeros(nn)
    d_u_K_numeric = np.zeros(nn)
    d_u_M_numeric = np.zeros(nn)

    for i in range(nn):
        G = 2*np.pi/aM[i]
        d_u_K_analytic[i] = fs.dist_up_KGK_an(G,M,momentum,V,phi)
        d_u_M_analytic[i] = fs.dist_up_MGM_an(G,M,momentum,V,phi)
        d_u_K_numeric[i] = fs.dist_up_KGK_num(G,M,momentum,V,phi)
        d_u_M_numeric[i] = fs.dist_up_MGM_num(G,M,momentum,V,phi)

########################################################################## Picture

fig,ax = plt.subplots()
fig.set_size_inches(10,8)
ax_r = ax.twinx()
s_ = 20

if 1:
    ax.plot(aM,d_e_K_analytic,'g',label=r'$K\Gamma K$')
    ax.plot(aM,d_e_M_analytic,'limegreen',label=r'$M\Gamma M$')
    ax.plot(aM,d_e_K_numeric,'g',ls='dashed',label=r'$K\Gamma K$ num')
    ax.plot(aM,d_e_M_numeric,'limegreen',ls='dashed',label=r'$M\Gamma M$ num')
    #
    ax.plot([aM[0],aM[-1]],[exp_dist_ext,exp_dist_ext],'aqua',label='exp S3')
    #
    ax.set_ylim(0,2*exp_dist_ext)
    ax.set_xlabel(r'$a_M$ ($\mathring{A}$)',size=s_)
    ax.set_ylabel(r'$\lambda$ ($\mathring{A}^{-1}$)',size=s_,rotation=0)
    ax.legend(fontsize=s_-10,loc='upper left')
if 1:
    ax_r.plot(aM,d_u_K_analytic,'firebrick',label=r'$K\Gamma K$')
    ax_r.plot(aM,d_u_M_analytic,'r',label=r'$M\Gamma M$')
    ax_r.plot(aM,d_u_K_numeric,'firebrick',ls='dashed',label=r'$K\Gamma K$ num')
    ax_r.plot(aM,d_u_M_numeric,'r',ls='dashed',label=r'$M\Gamma M$ num')
    #
    ax_r.set_ylim(0,2*exp_dist_up)
    ax_r.set_xlabel(r'$a_M$ ($\mathring{A}$)',size=s_)
    ax_r.set_ylabel(r'$\gamma$ ($eV$)',size=s_,rotation=0)
    ax_r.legend(fontsize=s_-10,loc='upper right')

plt.show()
