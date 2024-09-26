import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

a_WSe2 = 3.32   #Ang
a_WS2 = 3.18   #Ang

def dist_ext_KGK_an(g,m,e,v):
    E = abs(3*v**2/2/g**2-e)
    d0 = g-np.sqrt(2*m*E)+np.sqrt(2*m*E-g**2/3)
    d1 = -2*m*v/2/np.sqrt(2*m*E-g**2/3)
    d2 = -3*m*np.sqrt(2*m)*v**2/(2*g**2*np.sqrt(E)-3*np.sqrt(2*m)*g*E)
    return d0+d1+d2
def dist_up_KGK_an(g,m,k,v,phi):
    d0 = g*(3*k-2*g)/3/m
    d1 = -v
    d2 = 3*m*(np.cos(3*phi)-3)/g/(3*k-2*g)*v**2
    return d0+d1+d2

def dist_ext_MGM_an(g,m,e,v):
    E = abs(3*v**2/2/g**2-e)
    d0 = (g-np.sqrt(6*m*E)+np.sqrt(6*m*E-3*g**2))/2
    d2 = - 9*m*np.sqrt(2*m)*v**2/(4*g**2*np.sqrt(3*E)-3*np.sqrt(2*m)*g*E)
    return d0+d2
def dist_up_MGM_an(g,m,k,v,phi):
    d0 = 2*g*(k-g)/3/m
    d2 = 3*m/g/(k-g)*v**2
    return d0+d2

def Ham(v_k,g,m,v,phi,eta):
    V = v*np.exp(1j*phi)
    W = v*np.exp(-1j*phi)
    G1 = np.array([1,1/np.sqrt(3)])*g
    G1 = np.matmul(R_z(eta),G1)
    G2 = np.array([0,2/np.sqrt(3)])*g
    G2 = np.matmul(R_z(eta),G2)
    H = np.array([
        [-norm(v_k)**2/2/m,V,W,V,W,V,W],
        [W,-norm(v_k-G1)**2/2/m,V,0,0,0,V],
        [V,W,-norm(v_k-G2)**2/2/m,W,0,0,0],
        [W,0,V,-norm(v_k+G1-G2)**2/2/m,V,0,0],
        [V,0,0,W,-norm(v_k+G1)**2/2/m,W,0],
        [W,0,0,0,V,-norm(v_k+G2)**2/2/m,V],
        [V,W,0,0,0,W,-norm(v_k-G1+G2)**2/2/m],
        ])
    return H

def dist_ext_KGK_num(g,m,e,v,phi):
    nn = 1000
    list_k = np.linspace(2*g/3,2*np.pi/a_WSe2,nn)   #Start after the crossing of the bands
    bands = np.zeros((nn,2))
    for i in range(nn):
        H = Ham(np.array([list_k[i],0]),g,m,v,phi,0)
        bands[i] = np.linalg.eigvalsh(H)[-3:-1] #Two top energy bands are the relevant ones
    #indices
    E = np.linalg.eigvalsh(Ham(np.array([0,0]),g,m,v,phi,0))[-1]-e
    ind_mb = np.argmin(abs(bands[:,0]-E))
    ind_sb = np.argmin(abs(bands[:,1]-E))
    return list_k[ind_sb]-list_k[ind_mb]

def dist_ext_MGM_num(g,m,e,v,phi):
    nn = 1000
    list_k = np.linspace(g,2*np.pi/a_WSe2,nn)   #Start after the crossing of the bands
    bands = np.zeros((nn,2))
    for i in range(nn):
        H = Ham(np.array([list_k[i],list_k[i]/np.sqrt(3)]),g,m,v,phi,0)
        bands[i] = np.linalg.eigvalsh(H)[-3:-1] #Two top energy bands are the relevant ones
    #indices
    E = np.linalg.eigvalsh(Ham(np.array([0,0]),g,m,v,phi,0))[-1]-e
    ind_mb = np.argmin(abs(bands[:,0]-E))
    ind_sb = np.argmin(abs(bands[:,1]-E))
    return list_k[ind_sb]-list_k[ind_mb]

def dist_ext_rot(g,m,e,v,phi):
    nn = 3000
    eta = miniBZ_rotation(twist_angle(2*np.pi/g))
    list_k = np.linspace(g,2*np.pi/a_WSe2,nn)   #Start after the crossing of the bands
    bands = np.zeros((nn,7))
    we = np.zeros((nn,7))
    for i in range(nn):
        H = Ham(np.array([list_k[i],0]),g,m,v,phi,eta)
        bands[i],ev = np.linalg.eigh(H) 
        we[i] = np.abs(ev[0])**2
    #indices
    E = np.linalg.eigvalsh(Ham(np.array([0,0]),g,m,v,phi,eta))[-1]-e  #energy of the cut from the VBM
    ind_ks = np.argmin(abs(bands-E),axis=0) #k_points for all 7 bands
    ind_mb = np.argmax(np.array([we[ind_ks[i],i] for i in range(7)]))
    ind_kmb = ind_ks[ind_mb]
    if ind_mb>5:
        ind_sb1=ind_sb2=ind_mb
    elif ind_mb>4:
        ind_sb1=ind_sb2=ind_mb+1
    else:
        ind_sb1 = ind_mb+1
        ind_sb2 = ind_mb+2
    ind_ksb1 = ind_ks[ind_sb1]
    ind_ksb2 = ind_ks[ind_sb2]
    if 0:
        fig,ax = plt.subplots()
        fig.set_size_inches(15,15)
        for i in range(7):
            co = 'r' if i==ind_mb else 'b'
            ax.plot(list_k,bands[:,i],c='k',lw=0.3)
            ax.scatter(list_k,bands[:,i],s=we[:,i]*30,c=co,lw=0)
        ax.scatter(list_k[ind_kmb],bands[ind_kmb,ind_mb],s=30,c='g')
        ax.scatter(list_k[ind_ksb1],bands[ind_ksb1,ind_mb+1],s=30,c='g')
        ax.scatter(list_k[ind_ksb2],bands[ind_ksb2,ind_mb+2],s=30,c='g')
        ax.plot([g,list_k[-1]],[E,E],c='r')
        ax.set_ylim(2*E,5*np.max(bands[:,-1]))
        ax.set_title("{:.1f}".format(eta/np.pi*180))
        plt.show()
    return np.array([list_k[ind_ksb1]-list_k[ind_kmb],list_k[ind_ksb2]-list_k[ind_kmb]])

def dist_ext_rot_full(g,M,data,v,phi):
    nn = 3000
    eta = miniBZ_rotation(twist_angle(2*np.pi/g))
    list_k = np.linspace(g,2*np.pi/a_WSe2,nn)   #Start after the crossing of the bands
    result = np.zeros((2,2))
    for ss in range(2):
        s = '3' if ss==0 else '11'
        bands = np.zeros((nn,7))
        we = np.zeros((nn,7))
        for i in range(nn):
            H = Ham(np.array([list_k[i],0]),g,M[s],v,phi,eta)
            bands[i],ev = np.linalg.eigh(H)
            we[i] = np.abs(ev[0])**2
        #indices
        E = np.linalg.eigvalsh(Ham(np.array([0,0]),g,M[s],v,phi,eta))[-1]-data[s][0,0]  #energy of the cut from the VBM
        ind_ks = np.argmin(abs(bands-E),axis=0) #k_points for all 7 bands
        ind_mb = np.argmax(np.array([we[ind_ks[i],i] for i in range(7)]))
        ind_kmb = ind_ks[ind_mb]
        if ind_mb>5:
            ind_sb1=ind_sb2=ind_mb
        elif ind_mb>4:
            ind_sb1=ind_sb2=ind_mb+1
        else:
            ind_sb1 = ind_mb+1
            ind_sb2 = ind_mb+2
        ind_ksb1 = ind_ks[ind_sb1]
        ind_ksb2 = ind_ks[ind_sb2]
        result[ss] = np.array([list_k[ind_ksb1]-list_k[ind_kmb],list_k[ind_ksb2]-list_k[ind_kmb]])
    return result 

def dist_up_KGK_num(g,m,k,v,phi):
    H = Ham(np.array([k,0]),g,m,v,phi)
    bands = np.linalg.eigvalsh(H)[-3:-1] #Two top energy bands are the relevant ones
    return abs(bands[0]-bands[1])

def dist_up_MGM_num(g,m,k,v,phi):
    H = Ham(np.array([k,k/np.sqrt(3)]),g,m,v,phi)
    bands = np.linalg.eigvalsh(H)[-4:-2] #Two top energy bands are the relevant ones
    return abs(bands[0]-bands[1])


def twist_angle(a_m):
    return np.arccos(a_WSe2*a_WS2/2*(1/a_WSe2**2+1/a_WS2**2-1/a_m**2))

def moire_length(theta):
    return 1/np.sqrt(1/a_WSe2**2+1/a_WS2**2-2*np.cos(theta)/a_WSe2/a_WS2)

def R_z(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

A1 = np.array([1,0])
A2 = np.array([-1/2,np.sqrt(3)/2])
e_z = np.array([0, 0, 1])

def rep_ll(a1,a2):
    #Reciprocal vectors
    V = np.linalg.norm(np.cross(a1, a2))
    b1 = 2 * np.pi * np.cross(a2, e_z) / V
    b2 = 2 * np.pi * np.cross(e_z, a1) / V
    return b1[:2],b2[:2]

def miniBZ_rotation(theta):
    return np.arctan(-np.tan(theta/2)*(a_WSe2+a_WS2)/(a_WSe2-a_WS2))

