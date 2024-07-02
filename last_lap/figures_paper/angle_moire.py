import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

a_d = 3.32
a_u = 3.18
A1 = np.array([1,0])
A2 = np.array([-1/2,np.sqrt(3)/2])
e_z = np.array([0, 0, 1])
B1 = 2*np.pi*np.array([1,1/np.sqrt(3)])
B2 = 2*np.pi*np.array([0,2/np.sqrt(3)])

def a_M(theta):
    return 1/np.sqrt(1/a_u**2+1/a_d**2-2*np.cos(theta)/a_u/a_d)

def R_z(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

def rep_ll(a1,a2):
    #Reciprocal vectors
    V = np.linalg.norm(np.cross(a1, a2))
    b1 = 2 * np.pi * np.cross(a2, e_z) / V
    b2 = 2 * np.pi * np.cross(e_z, a1) / V
    return b1[:2],b2[:2]

nn = 3000
list_t = np.linspace(0,2*np.pi,nn)
phis = np.zeros(nn)
a_Ms = np.zeros(nn)
for i in range(nn):
    theta = list_t[i]
    A1u = np.matmul(R_z(theta/2),A1)*a_u
    A2u = np.matmul(R_z(theta/2),A2)*a_u
    B1u,B2u = rep_ll(A1u,A2u)
    A1d = np.matmul(R_z(-theta/2),A1)*a_d
    A2d = np.matmul(R_z(-theta/2),A2)*a_d
    B1d,B2d = rep_ll(A1d,A2d)
    #
    G1 = B1u-B1d
    phis[i] = (np.arctan2(G1[1],G1[0])-np.pi/6+1e-4)%(2*np.pi)
    a_Ms[i] = a_M(theta)
    #print(theta, A1u,A2u,B1u,B2u,a_Ms[i],phis[i]/180*np.pi)
    #input()

fig,ax = plt.subplots()

ax.plot(list_t,a_Ms,color='red')

ax_r = ax.twinx()
ax_r.plot(list_t,phis,color='green')

s_ = 20
ax.set_xlabel(r"$\theta$",size=s_)
ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],[r"$0$",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
ax.set_ylabel(r"$a_M$ ($\mathring{A}$)",size=s_)

ax_r.set_ylabel(r"$\varphi$",size=s_)
ax_r.set_yticks([phis[0],np.pi/2,np.pi],[r"$0$",r"$\pi/2$",r"$\pi$"])


inset_ax = inset_axes(ax, width="30%", height=1.5, loc="upper center")

#first 5 degrees in inset
n = int(nn/360*5)+2
inset_ax.plot(list_t[:n],a_Ms[:n],color='red')
inset_ax.set_xticks([0,list_t[np.argmin(abs(phis-np.pi/6))],list_t[int(nn/360*3)],list_t[n-1]],[r"$0$"+'°',"{:.1f}".format(list_t[np.argmin(abs(phis-np.pi/6))]/np.pi*180)+'°',"{:.1f}".format(list_t[int(nn/360*3)]/np.pi*180)+'°',"{:.1f}".format(list_t[n-1]/np.pi*180)+'°'])
inset_ax.plot([list_t[np.argmin(abs(phis-np.pi/6))],list_t[np.argmin(abs(phis-np.pi/6))]],[a_Ms[0],a_Ms[n-1]],color='gray',lw=0.5,ls='dashed')
#inset_ax.plot([list_t[int(nn/360*3)],list_t[int(nn/360*3)]],[a_Ms[0],a_Ms[n-1]],color='gray',lw=0.5,ls='dashed')
inset_ax_r = inset_ax.twinx()
inset_ax_r.plot(list_t[:n],phis[:n],color='green')
inset_ax_r.plot([0,list_t[n-1]],[np.pi/6,np.pi/6],color='gray',lw=0.5,ls='dashed')
inset_ax_r.set_yticks([phis[0],np.pi/6,phis[n-1]],["{:.1f}".format(phis[0]/np.pi*180)+'°',"{:.1f}".format(30)+'°',"{:.1f}".format(phis[n-1]/np.pi*180)+'°'])

plt.show()



