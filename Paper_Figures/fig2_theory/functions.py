import numpy as np
import scipy
import pickle

def compute_data(save_data,data_fn,*args):
    side_bands,momentum_points,V_list,aM_list,phi,mass,factor = args
    N = side_bands
    gaps = np.zeros((len(V_list),len(aM_list)))
    SBC = np.zeros((len(V_list),len(aM_list)))      #side band crossing
    h_ext = np.zeros((len(V_list),len(aM_list),2))    #the last dimension (2) is for displacement and relative weight
    h_int = np.zeros((len(V_list),len(aM_list),2))
    v_up = np.zeros((len(V_list),len(aM_list),2))
    v_dow = np.zeros((len(V_list),len(aM_list),2))
    list_momenta = np.zeros((len(V_list),len(aM_list),momentum_points))
    energies = np.zeros((len(V_list),len(aM_list),2*N+1,momentum_points))
    weights = np.zeros((len(V_list),len(aM_list),2*N+1,momentum_points))
    #
    for v in range(len(V_list)):
        for aM in range(len(aM_list)):
            G = np.pi*2/aM_list[aM]
            k_i = -4*G
            k_f = 4*G
            list_momenta[v,aM] = np.linspace(k_i,k_f,momentum_points)
            en = np.zeros((2*N+1,momentum_points))
            wp = np.zeros((2*N+1,momentum_points))
            for ii,k in enumerate(list_momenta[v,aM]):
                en[:,ii],ev = scipy.linalg.eigh(Hamiltonian(k,V_list[v],phi,N,G))
                wp[:,ii] = np.absolute(ev[N])**2
            energies[v,aM] = en
            weights[v,aM] = wp
            #
            gaps[v,aM] = gap(en,-G/2,list_momenta[v,aM])
            SBC[v,aM] = sbc(en,list_momenta[v,aM])
            #
            e_ = -(factor*G)**2/2/mass        #energy of main band between 2nd and third side band max
            l, inds = horizontal_displacement(e_,en,wp,list_momenta[v,aM])
            i_mb,i_sb1,i_sb2 = inds
            h_ext[v,aM,0] = abs(list_momenta[v,aM,l[i_sb1,1]]-list_momenta[v,aM,l[i_mb,1]])
            h_ext[v,aM,1] = wp[l[i_sb1,0],l[i_sb1,1]]/wp[l[i_mb,0],l[i_mb,1]]
            h_int[v,aM,0] = abs(list_momenta[v,aM,l[i_sb2,1]]-list_momenta[v,aM,l[i_mb,1]])
            h_int[v,aM,1] = wp[l[i_sb2,0],l[i_sb2,1]]/wp[l[i_mb,0],l[i_mb,1]]
            #
            i_k = l[i_mb,1]
            i_mb,i_sb1,i_sb2 = vertical_displacement(i_k,wp)
            v_up[v,aM,0] = abs(en[i_sb1,i_k]-en[i_mb,i_k])
            v_up[v,aM,1] = wp[i_sb1,i_k]/wp[i_mb,i_k]
            v_dow[v,aM,0] = abs(en[i_sb2,i_k]-en[i_mb,i_k])
            v_dow[v,aM,1] = wp[i_sb2,i_k]/wp[i_mb,i_k]
    data = {}
    data['gaps'] = gaps
    data['SBC'] = SBC
    data['h_out'] = h_ext
    data['h_inn'] = h_int
    data['v_upp'] = v_up
    data['v_low'] = v_dow
    data['momenta'] = list_momenta
    data['energies'] = energies
    data['weights'] = weights
    if save_data:
        with open(data_fn,'wb') as f:
            pickle.dump(data,f)
    return data

def Hamiltonian(k,V,phi,N,G,mass=1):
    """
    Simply the Hamiltonian, which has the dispersion in the diagonal entries (quadratic here for simplicity) and in the first diagonal the moire potential coupling.
    N -> # of side bands on each side
    G -> reciprocal moire lattice
    """
    H = np.zeros((2*N+1,2*N+1),dtype=complex)
    for n in range(-N,N):
        H[n+N,n+N] = -(k+n*G)**2/2/mass
        H[n+N,n+N+1] = V*np.exp(1j*phi)
        H[n+N+1,n+N] = V*np.exp(-1j*phi)
    H[2*N,2*N] = -(k+N*G)**2/2/mass
    return H

#Gap
def ind_k(k_pt,list_momenta):
    """
    Index of k_pt in momentum list.
    """
    initial_momentum = list_momenta[0]
    final_momentum = list_momenta[-1]
    momentum_points = len(list_momenta)
    return int(momentum_points*(k_pt-initial_momentum)/(final_momentum-initial_momentum))
def gap(energies,G_pt,list_momenta):
    """
    Energy gap between two neighboring bands.
    level specifies which bands from the top are considered.
    G_pt specifies in which momentum the gap is considered.
    """
    return (energies[-1]-energies[-2])[ind_k(G_pt,list_momenta)]

def sbc(energies,list_momenta):
    return (energies[-1]-energies[-2])[ind_k(0,list_momenta)]

#band distance
def horizontal_displacement(e_,energies,weights,list_momenta,mass=1):
    """
    Compute indeces of main and first two side bands, using the weights to discriminate.
    """
    momentum_points = len(list_momenta)
    delta_e = 1/2/mass*np.sqrt(2*mass*abs(e_))*(list_momenta[1]-list_momenta[0])
    l = np.argwhere(abs(e_-energies[:,:momentum_points//2])<delta_e)
    indices = np.argsort([weights[l[i,0],l[i,1]] for i in range(l.shape[0])])
    filtered_indices = [indices[-1],]
    for i in range(indices.shape[0]):
        temp = indices[-1-i]
        if abs(l[temp,1]-l[filtered_indices[-1],1])>2:
                filtered_indices.append(temp)
    i_mb = filtered_indices[0]
    i_sb1 = filtered_indices[1]
    i_sb2 = filtered_indices[2]
    return l,(i_mb,i_sb1,i_sb2)
def vertical_displacement(i_k,weights):
    indices = np.argsort(weights[:,i_k])
    i_mb = indices[-1]
    i_sb2 = indices[-2]
    i_sb1 = indices[-3] if indices[-3] < indices[-2] else indices[-4]
    return (i_mb,i_sb1,i_sb2)

def get_data_fn(*args):
    fn = 'data/'
    for i in range(len(args)):
        p = args[i]
        if type(p) in [int,np.int64]:
            fn += str(p)
        elif type(p) in [float,np.float64]:
            fn += "{:.5f}".format(p)
        if i != len(args):
            fn += '_'
    fn += '.pkl'
    return fn
