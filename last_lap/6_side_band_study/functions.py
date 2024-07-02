import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

a_WSe2 = 3.32   #Ang

def dist_ext_MGM_an(g,m,e,v):
    d0 = (g-np.sqrt(6*m*abs(e))+np.sqrt(6*m*abs(e)-3*g**2))/2
    d2 = - 9*m*np.sqrt(2*m)*v**2/(4*g**2*np.sqrt(3*abs(e))-3*np.sqrt(2*m)*g*abs(e))
    return d0+d2
def dist_up_MGM_an(g,m,k,v,phi):
    d0 = 2*g*(k-g)/3/m
    d2 = 3*m/g/(k-g)*v**2
    return d0+d2
def dist_dw_MGM_an(g,m,k,v):
    d0 = 2/3/m*g*(k+g)
    d2 = 3*m/g/(k+g)*v**2
    return d0+d2

def dist_ext_KGK_an(g,m,e,v):
    E = abs(e)
    d0 = g-np.sqrt(2*m*E)+np.sqrt(2*m*E-g**2/3)
    d2 = -3*m*np.sqrt(2*m)*v**2/(2*g**2*np.sqrt(E)-3*np.sqrt(2*m)*g*E)
    return d0+d2
def dist_up_KGK_an(g,m,k,v,phi):
    d0 = g*(3*k-2*g)/3/m
    d1 = -v
    d2 = 3*m*(np.cos(3*phi)-3)/g/(3*k-2*g)*v**2
    return d0+d1+d2
def dist_dw_KGK_an(g,m,k,v):
    d0 = 2*g**2/3/m
    d2 = 3*m*v**2/g**2
    return d0+d2

def Ham(v_k,g,m,v,phi):
    V = v*np.exp(1j*phi)
    W = v*np.exp(-1j*phi)
    G1 = np.array([1,1/np.sqrt(3)])*g
    G2 = np.array([0,2/np.sqrt(3)])*g
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
        H = Ham(np.array([list_k[i],0]),g,m,v,phi)
        bands[i] = np.linalg.eigvalsh(H)[-3:-1] #Two top energy bands are the relevant ones
    #indices
    ind_mb = np.argmin(abs(bands[:,0]-e))
    ind_sb = np.argmin(abs(bands[:,1]-e))
    return list_k[ind_sb]-list_k[ind_mb]

def dist_ext_MGM_num(g,m,e,v,phi):
    nn = 1000
    list_k = np.linspace(g,2*np.pi/a_WSe2,nn)   #Start after the crossing of the bands
    bands = np.zeros((nn,2))
    for i in range(nn):
        H = Ham(np.array([list_k[i],list_k[i]/np.sqrt(3)]),g,m,v,phi)
        bands[i] = np.linalg.eigvalsh(H)[-4:-2] #Two top energy bands are the relevant ones
    #indices
    ind_mb = np.argmin(abs(bands[:,0]-e))
    ind_sb = np.argmin(abs(bands[:,1]-e))
    return list_k[ind_sb]-list_k[ind_mb]

def dist_up_KGK_num(g,m,k,v,phi):
    H = Ham(np.array([k,0]),g,m,v,phi)
    bands = np.linalg.eigvalsh(H)[-3:-1] #Two top energy bands are the relevant ones
    return abs(bands[0]-bands[1])

def dist_up_MGM_num(g,m,k,v,phi):
    H = Ham(np.array([k,k/np.sqrt(3)]),g,m,v,phi)
    bands = np.linalg.eigvalsh(H)[-4:-2] #Two top energy bands are the relevant ones
    return abs(bands[0]-bands[1])





