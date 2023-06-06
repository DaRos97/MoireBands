import numpy as np
import matplotlib.pyplot as plt

a_1 = np.array([1,0])
a_2 = np.array([-1/2,np.sqrt(3)/2])
J_plus = ((3,5), (6,8), (9,11))
J_minus = ((1,2), (3,4), (4,5), (6,7), (7,8), (9,10), (10,11))
J_MX_plus = ((3,1), (5,1), (4,2), (10,6), (9,7), (11,7), (10,8))
J_MX_minus = ((4,1), (3,2), (5,2), (9,6), (11,6), (10,7), (9,8), (11,8))
#####
#####Hamiltonian part
#####
#Here I contruct the single 11 orbital hamiltonian, which is 22x22 for spin-orbit, as a function of momentum
#Detail can be found in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.205108
def H_monolayer(K_p,hopping,epsilon,HSO,a_mono):
    t = hopping
    k_x,k_y = K_p       #momentum
    delta = a_mono* np.array([a_1, a_1+a_2, a_2, -(2*a_1+a_2)/3, (a_1+2*a_2)/3, (a_1-a_2)/3, -2*(a_1+2*a_2)/3, 2*(2*a_1+a_2)/3, 2*(a_2-a_1)/3])
    H_0 = np.zeros((11,11),dtype=complex)       #fist part without SO
    #Diagonal
    for i in range(11):
        H_0[i,i] += (epsilon[i] + 2*t[0][i,i]*np.cos(np.dot(K_p,delta[0])) 
                             + 2*t[1][i,i]*(np.cos(np.dot(K_p,delta[1])) + np.cos(np.dot(K_p,delta[2])))
                 )
    #Off diagonal symmetry +
    for ind in J_plus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (2*t[0][i,j]*np.cos(np.dot(K_p,delta[0])) 
                + t[1][i,j]*(np.exp(-1j*np.dot(K_p,delta[1])) + np.exp(-1j*np.dot(K_p,delta[2])))
                + t[2][i,j]*(np.exp(1j*np.dot(K_p,delta[1])) + np.exp(1j*np.dot(K_p,delta[2])))
                )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #Off diagonal symmetry -
    for ind in J_minus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (-2*1j*t[0][i,j]*np.sin(np.dot(K_p,delta[0])) 
                + t[1][i,j]*(np.exp(-1j*np.dot(K_p,delta[1])) - np.exp(-1j*np.dot(K_p,delta[2])))
                + t[2][i,j]*(-np.exp(1j*np.dot(K_p,delta[1])) + np.exp(1j*np.dot(K_p,delta[2])))
                )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #M-X coupling +
    for ind in J_MX_plus:
        i = ind[0]-1
        j = ind[1]-1
        temp = t[3][i,j] * (np.exp(1j*np.dot(K_p,delta[3])) - np.exp(1j*np.dot(K_p,delta[5])))
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #M-X coupling -
    for ind in J_MX_minus:
        i = ind[0]-1
        j = ind[1]-1
        temp = (t[3][i,j] * (np.exp(1j*np.dot(K_p,delta[3])) + np.exp(1j*np.dot(K_p,delta[5])))
                   + t[4][i,j] * np.exp(1j*np.dot(K_p,delta[4]))
                   )
        H_0[i,j] += temp
        H_0[j,i] += np.conjugate(temp)
    #Second nearest neighbor
    H_1 = np.zeros((11,11),dtype=complex)       #fist part without SO
    H_1[8,5] += t[5][8,5]*(np.exp(1j*np.dot(K_p,delta[6])) + np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,8] += np.conjugate(H_1[8,5])
    #
    H_1[10,5] += t[5][10,5]*(np.exp(1j*np.dot(K_p,delta[6])) - 1/2*np.exp(1j*np.dot(K_p,delta[7])) - 1/2*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,10] += np.conjugate(H_1[10,5])
    #
    H_1[9,5] += np.sqrt(3)/2*t[5][10,5]*(- np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[5,9] += np.conjugate(H_1[9,5])
    #
    H_1[8,7] += t[5][8,7]*(np.exp(1j*np.dot(K_p,delta[6])) - 1/2*np.exp(1j*np.dot(K_p,delta[7])) - 1/2*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[7,8] += np.conjugate(H_1[8,7])
    #
    H_1[8,6] += np.sqrt(3)/2*t[5][8,7]*(- np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,8] += np.conjugate(H_1[8,6])
    #
    H_1[9,6] += 3/4*t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[7])) + np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,9] += np.conjugate(H_1[9,6])
    #
    H_1[10,6] += np.sqrt(3)/4*t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[7])) - np.exp(1j*np.dot(K_p,delta[8])))
    H_1[6,10] += np.conjugate(H_1[10,6])
    H_1[9,7] += H_1[10,6]
    H_1[7,9] += H_1[6,10]
    #
    H_1[10,7] += t[5][10,7]*(np.exp(1j*np.dot(K_p,delta[6])) + 1/4*np.exp(1j*np.dot(K_p,delta[7])) + 1/4*np.exp(1j*np.dot(K_p,delta[8])))
    H_1[7,10] += np.conjugate(H_1[10,7])
    #Combine the two terms
    H_TB = H_0 + H_1

    #### Spin orbit terms
    H = np.zeros((22,22),dtype = complex)
    H[:11,:11] = H_TB
    H[11:,11:] = H_TB
    H += HSO
    return H

#####
#####Moire potential part
#####
#Here I construct the single Moire potential parts. Function of the reciprocal Moire vector -> g=0,..,5
#It is diagonal since it does not mix the bands and/or spins. 
#Different values for in-plane (indexes 1,2) and out-of-plane (index 0) orbits.
def V_g(g,params):          #g is a integer from 0 to 5
    V_G,psi_G,V_K,psi_K = params        #extract the parameters
    Id = np.zeros((22,22),dtype = complex)
    out_of_plane = V_G*np.exp(1j*(-1)**(g%2)*psi_G)
    in_plane = V_K*np.exp(1j*(-1)**(g%2)*psi_K)
    list_out = (0,1,2,5,8)
    list_in = (3,4,6,7,9,10)
    for i in list_out:
        Id[i,i] = Id[i+11,i+11] = out_of_plane
    for i in list_in:
        Id[i,i] = Id[i+11,i+11] = in_plane
    return Id

#####
#####Total Giga-Hamiltonian
#####
#Here I put all together to form the giga-Hamiltonian matrix by considering momentum space neighbors up to the cutoff N
#To do it I give a number to each of the mini BZs, starting from the central one and then one circle at a time, going anticlockwise.
#In the look-up table "lu" I put the positions of these mini-BZs, using coordinates in units of G1=(1,0), G2=(1/2,sqrt(3)/2). 
def total_H(K_,N,hopping,epsilon,HSO,params_V,G_M,a_mono):
    #Dimension of the Hamiltonian, which is 22 (11 bands + SO) times the number of mini-BZs. 
    #For N=0 the number of mini-BZ is 1, for N=1 it is 6+1, for N=2 it is 12+6+1 ecc..
    n_cells = int(1+3*N*(N+1))*22
    H = np.zeros((n_cells,n_cells),dtype=complex)
    ############
    ############ Diagonal (monolayer) part and look-up table of site's coordinates
    ############
    #Look up matrix of coordinates of single mini-BZs, in units of G1=(1,0), G2=(1/2,sqrt(3)/2)
    lu = []     
    #Matrix "m" gives the increment to the coordinates to go around the circle, strating from 
    #the rightmost mini-BZ and going counter-clockwise.
    m = [[-1,1],[-1,0],[0,-1],[1,-1],[1,0],[0,1]]
    for n in range(0,N+1):      #circles go from 0 (central BZ) to N included
        i = 0
        j = 0
        #loop over indexes of mini-BZs in the circle
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       
            if s == np.sign(n)*(1+(n-1)*n*3):
                #first mini-BZ, which is the rightmost of the ring
                lu.append((n,0))           
            else:
                #Append the new mini-BZ by incrementing the previous with "m". 
                #Each value of m has to be repeated n times (each edge of exagon), which are counted by j. 
                lu.append((lu[-1][0]+m[i][0],lu[-1][1]+m[i][1]))
                if j == n-1:
                    i += 1
                    j = 0
                else:
                    j += 1
            #Take the correct momentum by adding the components of "lu" times the reciprocal 
            #lattice vectors to the initial momentum K_
            K_p = K_ + G_M[0]*lu[-1][0] + G_M[1]*lu[-1][1]
            #place the corresponding 22x22 Hamiltonian in its position
            H[s*22:s*22+22,s*22:s*22+22] = H_monolayer(K_p,hopping,epsilon,HSO,a_mono)
    
    ############
    ############ Off diagonal part --> Moire potential. 
    ############ We are placing a Moiré 22x22 potential just in the off-diagonal parts connected by a 
    ############ reciprocal lattice vector.
    ############
    for n in range(0,N+1):      #Circles
        for s in range(np.sign(n)*(1+(n-1)*n*3),n*(n+1)*3+1):       #Indices inside the circle
            #loop over the reciprocal directions
            for i in m:
                #Coordinates of the considered mini-BZ
                ind_s = lu[s]
                #Coordinames of the mini-BZ in the i direction wrt the previous
                ind_nn = (ind_s[0]+i[0],ind_s[1]+i[1])
                try:
                    #if the obtained indices are in "lu" (may not be because the external mini-BZs don't have all 
                    #the neighbors) ...->
                    nn = lu.index(ind_nn)
                except:
                    continue
                #index telling which reciprocal direction it is
                g = (m.index(i) + 2)%6
                #...-> add the Moiré potential in the giga-Hamiltonian
                H[s*22:(s+1)*22,nn*22:(nn+1)*22] = V_g(g,params_V)
    return H

#Lorentzian
def lorentzian_weight(k,e,*pars):
    K2,E2,weight,K_,E_ = pars
    return abs(weight)/((k-K_)**2+K2)/((e-E_)**2+E2)
#Banana Lorentzian
def banana_lorentzian_weight(kx_list,ky_list,*pars):
    Kx2,Ky2,E2,weight,E_,E_cut,Kx_,Ky_ = pars
    return abs(weight)/((kx_list-Kx_)**2+Kx2)/((ky_list-Ky_)**2+Ky2)/((E_-E_cut)**2+E2)

#z rotations
def R_z(t):
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R

#path in BZ
def pathBZ(path_name,a_monolayer,pts_ps):
    #Reciprocal lattice vectors with a. Starts from G[0] along y and goes counter-clockwise with R_6 rotations.
    #BZ has shape of hexagon with flat edges up and down
    G = [4*np.pi/np.sqrt(3)/a_monolayer*np.array([0,1])]      
    for i in range(1,6):
        G.append(np.tensordot(R_z(np.pi/3*i),G[0],1))
    #
    K = np.array([G[-1][0]/3*2,0])                      #K-point
    Gamma = np.array([0,0])                                #Gamma
    K2 =    K/2                             #Half is denoted by a '2'
    K2_ =   - K2                            #Opposite wrt G is denoted by '_'
    M =     G[-1]/2                          #M-point
    M_ =     - M
    M2 =    M/2 
    M2_ =   - M2
    Kp =    np.tensordot(R_z(np.pi/3),K,1)     #K'-point
    Kp_ = - Kp
    Kp2 =   Kp/2
    Kp2_ =   -Kp2
    dic_names = {'G':Gamma,
                 'K':K,
                 'M':M,
                 'm':M_,
                 'C':Kp, 
                 'c':Kp_, 
                 'Q':K2,
                 'q':K2_,
                 'N':M2,
                 'n':M2_,
                 'P':Kp2, 
                 'p':Kp2_,
                 }
    #plt.figure()
    #for p in dic_names.keys():
    #    plt.scatter(dic_names[p][0],dic_names[p][1],label=p)
    #plt.legend()
    #plt.show()
    path = []
    for i in range(len([*path_name])-1):
        Pi = dic_names[path_name[i]]
        Pf = dic_names[path_name[i+1]]
        direction = Pf-Pi
        for i in range(pts_ps):
            path.append(Pi+direction*i/pts_ps)
    K_points = []
    for i in [*path_name]:
        K_points.append(dic_names[i])
    return path, K_points

def gridBZ(grid_pars,a_monolayer):
    K_center,dist_kx,dist_ky,n_bands,pts_per_direction = grid_pars
    #K_center: string with name of central point of the grid
    #dist_k*: float of distance from central point of furthest point in each direction *
    #pts_per_direction: array of 2 floats with TOTAL number of steps in the two directions -> better if odd so central point is included
    G = [4*np.pi/np.sqrt(3)/a_monolayer*np.array([0,1]),]      
    for i in range(1,6):
        G.append(np.tensordot(R_z(np.pi/3*i),G[0],1))
    K = np.array([G[-1][0]/3*2,0])                      #K-point
    Gamma = np.array([0,0])                             #Gamma
    Kp =    np.tensordot(R_z(np.pi/3),K,1)              #K'-point
    dic_symm_pts = {'G':Gamma,'K':K,'C':Kp}
    #
    grid = np.zeros((pts_per_direction[0],pts_per_direction[1],2))
    KKK = dic_symm_pts[K_center]
    for x in range(-pts_per_direction[0]//2+1,pts_per_direction[0]//2+1):
        for y in range(-pts_per_direction[1]//2+1,pts_per_direction[1]//2+1):
            K_pt_x = KKK[0] + 2*dist_kx*x/pts_per_direction[0]
            K_pt_y = KKK[1] + 2*dist_ky*y/pts_per_direction[1]
            grid[x+pts_per_direction[0]//2,y+pts_per_direction[1]//2,0] = K_pt_x
            grid[x+pts_per_direction[0]//2,y+pts_per_direction[1]//2,1] = K_pt_y
    return grid

def get_Moire(a_M):     #Compute Moire recipèrocal lattice vectors
    G_M = [4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
    for i in range(1,6):
        G_M.append(np.tensordot(R_z(np.pi/3*i),G_M[0],1))
    return G_M

def tqdm(n):
    return n
