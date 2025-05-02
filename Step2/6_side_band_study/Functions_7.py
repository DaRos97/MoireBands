import numpy as np
import matplotlib.pyplot as plt


def compute_sb_distance(th,aV,n_sb):
    a1 = 3.32   #WSe2
    a2 = 3.18   #WS2
    aM = 1/np.sqrt(1/a1**2+1/a2**2-2*np.cos(th)/a1/a2)
    v1 = np.array([aM,0])
    v2 = np.array([-aM/2,aM/2*np.sqrt(3)])
    phi = 0
    G1,G2 = rep_ll(v1,v2)
    G1 = R_z(miniBZ_rotation(th,a1,a2))@G1
    G2 = R_z(miniBZ_rotation(th,a1,a2))@G2
    V = aV*np.exp(1j*phi)
    V_ = V.conj()
    Nk = 100
    ens = np.zeros((Nk,7))
    evs = np.zeros((Nk,7,7),dtype=complex)
    list_k = np.zeros((Nk,2))
    list_k[:,0] = np.linspace(-2,0,Nk)
    for ik in range(Nk):
        k = list_k[ik]
        H = np.array([[ham(k),V,V_,V,V_,V,V_],
                    [V,ham(k-G1),V,0,0,0,V],
                    [V,V_,ham(k-G2),V_,0,0,0],
                    [V_,0,V,ham(k+G1-G2),V,0,0],
                    [V,0,0,V_,ham(k+G1),V_,0],
                    [V_,0,0,0,V,ham(k+G2),V],
                    [V,V_,0,0,0,V,ham(k-G1+G2)]])
        ens[ik],evs[ik] = np.linalg.eigh(H)
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(7):
        ax.plot(list_k[:,0],ens[:,i])
    plt.show()
    exit()

def ham(k):
    mass = 1
    return -np.linalg.norm(k)**2/2/mass

e_z = np.array([0,0,1])
def rep_ll(a1,a2):
    #Reciprocal vectors
    V = np.linalg.norm(np.cross(a1, a2))
    b1 = 2 * np.pi * np.cross(a2, e_z) / V
    b2 = 2 * np.pi * np.cross(e_z, a1) / V
    return b1[:2],b2[:2]

def R_z(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
def miniBZ_rotation(theta,a1,a2):
    return np.arctan(-np.tan(theta/2)*(a1+a2)/(a1-a2))
