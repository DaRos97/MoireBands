import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Here we plot the interlayer coupling.
"""

if len(sys.argv)!=2:
    print("Usage: python3 plot_interlayer.py arg1")
    print("arg1 -> 0 or 1 for single or multiple orbitals")
    exit()
else:
    int_type = int(sys.argv[1])

"""
For single orbital we keep the maximum at r=0.
For multiple orbitals we shift one of the two to have maximum at one of the unit cell corners.
"""

aM = 80     #Angstrom
ptsx = 200
ptsy = 210
W1p = 0
W2p = 0.3

W1d = 1
W2d = 0.1

psip = np.pi/3
psid = 0*2*np.pi/3

if int_type==0:
    print("Plotting interlayer just one type of orbitals, with parameters")
    print("W1 = "+"{:.3f}".format(W1p))
    print("W2 = "+"{:.3f}".format(W2p))
    print("phase: "+"{:.3f}".format(psip))
elif int_type==1:
    print("Plotting interlayer with two types of orbitals, with parameters")
    print("W1p = "+"{:.3f}".format(W1p))
    print("W2p = "+"{:.3f}".format(W2p))
    print("phase_p: "+"{:.3f}".format(psip))
    print("W1d = "+"{:.3f}".format(W1d))
    print("W2d = "+"{:.3f}".format(W2d))
    print("phase_d: "+"{:.3f}".format(psid))
else:
    print("Unrecognized input: ",sys.argv[1])

a1 = aM*np.array([1,0])
a2 = aM*np.array([1/2,np.sqrt(3)/2])
a3 = aM*np.array([-1/2,np.sqrt(3)/2])

G1 = 4*np.pi/np.sqrt(3)/aM*np.array([np.sqrt(3)/2,1/2])
G2 = 4*np.pi/np.sqrt(3)/aM*np.array([0,1])
G3 = 4*np.pi/np.sqrt(3)/aM*np.array([-np.sqrt(3)/2,1/2])
G4 = -G1
G5 = -G2
G6 = -G3

def interlayer(rx,ry,w1,w2,psi):
    r = np.array([rx,ry])
    t1 = np.exp(-1j*(np.dot(G1,r)+psi))
    t2 = np.exp(-1j*(np.dot(G2,r)-psi))
    t3 = np.exp(-1j*(np.dot(G3,r)+psi))
    t4 = np.exp(-1j*(np.dot(G4,r)-psi))
    t5 = np.exp(-1j*(np.dot(G5,r)+psi))
    t6 = np.exp(-1j*(np.dot(G6,r)-psi))
    return w1 + w2*(t1+t2+t3+t4+t5+t6)

fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot()

x_list = np.linspace(-a1[0],a1[0],ptsx)
y_list = np.linspace(-(a2+a3)[1]/2,(a2+a3)[1]/2,ptsy)

if int_type==0:
    data = np.zeros((ptsx,ptsy))
    for ix in range(ptsx):
        for iy in range(ptsy):
            data[ix,iy] = np.real(interlayer(x_list[ix],y_list[iy],W1p,W2p,psip))
elif int_type==1:
    data = np.zeros((ptsx,ptsy))
    for ix in range(ptsx):
        for iy in range(ptsy):
            data[ix,iy] = np.real(interlayer(x_list[ix],y_list[iy],W1p,W2p,psip)) + np.real(interlayer(x_list[ix],y_list[iy],W1d,W2d,psid))

X,Y = np.meshgrid(x_list,y_list)

mesh = ax.pcolormesh(X,Y,data.T,cmap='plasma')
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Interlayer",fontsize=30)
cbar.ax.tick_params(labelsize=30)

#Hexagon
radius = np.linalg.norm((a1+a2)/3)
angles = np.linspace(np.pi/6, 2 * np.pi+np.pi/6, 7)
centers = [np.zeros(2), a1, a2, a3, -a1, -a2, -a3]
for i in range(7):
    x = centers[i][0] + radius * np.cos(angles)
    y = centers[i][1] + radius * np.sin(angles)
    ax.plot(x, y, 'k-',lw=2)

#ax.set_title("w1="+"{:.2f}".format(W1p)+' eV, w2='+"{:.3f}".format(W2p)+' eV',size=30)

ax.set_aspect('equal')
ax.set_xlim(x_list[0],x_list[-1])
ax.set_ylim(y_list[0],y_list[-1])
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()

if int_type==0:
    figname = "interlayer_"+"{:.3f}".format(psip)+".png"
elif int_type==1:
    figname = "interlayer_"+"{:.3f}".format(psip)+"_"+"{:.3f}".format(psid)+".png"

plt.savefig(figname)
plt.show()
