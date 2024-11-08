import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])
x = np.array([1,0])
y = np.array([0,1])

a_WSe2 = 1.1
a_WS2 = 1.
a_M = 1/np.sqrt(1/a_WSe2**2+1/a_WS2**2-2/a_WSe2/a_WS2)

print("moire length: ",a_M)

nx = ny = int(a_M/a_WSe2)+10
l_WSe2 = np.zeros((2*nx,2*ny,2,2))
l_WS2 = np.zeros((2*nx,2*ny,2,2))

#Start from the center
for i in range(-nx,nx):
    for j in range(-ny,ny):
        l_WSe2[i+nx,j+ny,0] = (i*a1+j*a2+x/2-y/2/np.sqrt(3))*a_WSe2
        l_WSe2[i+nx,j+ny,1] = (i*a1+j*a2+x/2+y/2/np.sqrt(3))*a_WSe2
        l_WS2[i+nx,j+ny,0] = (i*a1+j*a2+x/2-y/2/np.sqrt(3))*a_WS2
        l_WS2[i+nx,j+ny,1] = (i*a1+j*a2+x/2+y/2/np.sqrt(3))*a_WS2

fig= plt.figure(figsize=(20,10))
shapes = [['o','^'],['^','o']]
titles = ['Parallel','Anti-parallel']
for i in range(2):
    ax = fig.add_subplot(2,2,2*i+1)
    ax.set_aspect('equal')
    #
    shape1 = ['o','^']
    shape2 = shapes[i]
    for y in range(-ny,ny):
        for n in range(2):      #sublattice index
            ax.scatter(l_WSe2[:,y+ny,n,0],l_WSe2[:,y+ny,n,1],color='b',marker=shape1[n],lw=1,facecolor='none')
            ax.scatter(l_WS2[:,y+ny,n,0],l_WS2[:,y+ny,n,1],color='r',marker=shape2[n],lw=1,facecolor='none')
        for x in range(-nx,nx):
            x1,y1 = (x*a1+y*a2)*a_WSe2
            x2,y2 = (x*a1+y*a2)*a_WS2
            r1 = a_WSe2/np.sqrt(3)
            r2 = a_WS2/np.sqrt(3)
            hexagon1 = patches.RegularPolygon((x1,y1), numVertices=6, radius=r1, orientation=0, edgecolor='b', facecolor='none', zorder=0, lw=0.2)
            hexagon2 = patches.RegularPolygon((x2,y2), numVertices=6, radius=r2, orientation=0, edgecolor='r', facecolor='none', zorder=0, lw=0.2)
            ax.add_patch(hexagon1)
            ax.add_patch(hexagon2)
    #UC
    if 0:
        ax.arrow(0,0,a1[0]*a_M,a1[1]*a_M,color='k',head_width=0.,zorder=0,lw=0.5)
        ax.arrow(0,0,a2[0]*a_M,a2[1]*a_M,color='k',head_width=0.,zorder=0,lw=0.5)
        ax.arrow(a1[0]*a_M,a1[1]*a_M,a2[0]*a_M,a2[1]*a_M,color='k',head_width=0.,zorder=0,lw=0.5)
        ax.arrow(a2[0]*a_M,a2[1]*a_M,a1[0]*a_M,a1[1]*a_M,color='k',head_width=0.,zorder=0,lw=0.5)
    else:
        hexagon = patches.RegularPolygon((0,0), numVertices=6, radius=a_M/np.sqrt(3), orientation=0, edgecolor='k', facecolor='none', zorder=2, lw=2)
        ax.add_patch(hexagon)
    #
    ax.axis('off')

#    ax.set_xlim(-1.2*a_M/2,1.1*a_M)
#    ax.set_ylim(-a_WSe2*1.5,1.2*a_M/2*np.sqrt(3))
    ax.set_xlim(-a_M,a_M)
    ax.set_ylim(-a_M,a_M)
#
    ax.set_title(titles[i],size=20)
fig.tight_layout()
plt.show()
