import numpy as np
import matplotlib.pyplot as plt

a_d = 3.32
a_u = 3.18
A1 = np.array([1,0])
A2 = np.array([-1/2,np.sqrt(3)/2])
e_z = np.array([0, 0, 1])
B1 = 2*np.pi*np.array([1,1/np.sqrt(3)])
B2 = 2*np.pi*np.array([0,2/np.sqrt(3)])

def aM(th):
    return a_d*a_u/np.sqrt(a_d**2+a_u**2-2*a_d*a_u*np.cos(th))
def eta(th):
    return np.arctan(-np.tan(th/2)*(a_u+a_d)/(a_u-a_d))/np.pi*180

l_th = np.linspace(-5/180*np.pi,5/180*np.pi,1001)

s_ = 40

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot()  #left: aM
col1 = 'darkorange'
ax.plot(l_th,aM(l_th),color=col1)
ax.set_yticks(ax.get_yticks(),
                ["{:.0f}".format(ax.get_yticks()[i])+'°' for i in range(len(ax.get_yticks()))],
                size=s_,
                color=col1
               )
ax.set_ylabel(r"$a_M(\mathring{A})$",size=s_,color=col1)

ax_r = ax.twinx()   #right: eta
col2 = 'navy'
ax_r.plot(l_th,eta(l_th),color=col2)
ax_r.set_yticks(ax_r.get_yticks(),
                ["{:.0f}".format(ax_r.get_yticks()[i])+'°' for i in range(len(ax_r.get_yticks()))],
                size=s_,
                color=col2
               )
#for tick_label in ax_r.get_yticklabels():
#    tick_label.set_color('blue')
ax_r.set_ylabel(r"$\eta$",size=s_,color=col2)


n_th = 5
xticks = np.linspace(l_th[0],l_th[-1],n_th)
ax.set_xticks(xticks,["{:.1f}".format(xticks[i]/np.pi*180)+'°' for i in range(n_th)],size=s_)
ax.set_xlabel(r"$\theta$",size=s_)

plt.tight_layout()
plt.savefig('moire_angle.png')
plt.show()
