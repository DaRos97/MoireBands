import numpy as np
import matplotlib.pyplot as plt

#####
#####Hamiltonian part
#####
#Here I contruct the single 3 orbital hamiltonian, which is 6x6 for spin-orbit, as a function of momentum
#Detail can be found in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.085433
def H_p(P,params):
    k_x,k_y = P				#momentum
    lattice_constant,z_xx,e_1,e_2,t_0,t_1,t_2,t_11,t_12,t_22,r_0,r_1,r_2,r_11,r_12,u_0,u_1,u_2,u_11,u_12,u_22,lamb = params	#extract the parameters
    a = k_x*lattice_constant/2              #alpha
    b = k_y*lattice_constant*np.sqrt(3)/2   #beta
    V_0 = (e_1 + 2*t_0*(2*np.cos(a)*np.cos(b)+np.cos(2*a)) 
                + 2*r_0*(2*np.cos(3*a)*np.cos(b)+np.cos(2*b))
           +2*u_0*(2*np.cos(2*a)*np.cos(2*b)+np.cos(4*a))
           )
    V_1 = complex(-2*np.sqrt(3)*t_2*np.sin(a)*np.sin(b)
                  +2*(r_1+r_2)*np.sin(3*a)*np.sin(b)
                  -2*np.sqrt(3)*u_2*np.sin(2*a)*np.sin(2*b),
                  2*t_1*np.sin(a)*(2*np.cos(a)+np.cos(b))
                  +2*(r_1-r_2)*np.sin(3*a)*np.cos(b)
                  +2*u_1*np.sin(2*a)*(2*np.cos(2*a)+np.cos(2*b))
                )
    V_2 = complex(2*t_2*(np.cos(2*a)-np.cos(a)*np.cos(b))
                  -2/np.sqrt(3)*(r_1+r_2)*(np.cos(3*a)*np.cos(b)-np.cos(2*b))
                  +2*u_2*(np.cos(4*a)-np.cos(2*a)*np.cos(2*b)),
                  2*np.sqrt(3)*t_1*np.cos(a)*np.sin(b)
                  +2/np.sqrt(3)*np.sin(b)*(r_1-r_2)*(np.cos(3*a)+2*np.cos(b))
                  +2*np.sqrt(3)*u_1*np.cos(2*a)*np.sin(2*b)
                )
    V_11 = (e_2 + (t_11+3*t_22)*np.cos(a)*np.cos(b) + 2*t_11*np.cos(2*a)
            +4*r_11*np.cos(3*a)*np.cos(b) + 2*(r_11+np.sqrt(3)*r_12)*np.cos(2*b)
            +(u_11+3*u_22)*np.cos(2*a)*np.cos(2*b) + 2*u_11*np.cos(4*a)
            )
    V_12 = complex(np.sqrt(3)*(t_22-t_11)*np.sin(a)*np.sin(b) + 4*r_12*np.sin(3*a)*np.sin(b)
                   +np.sqrt(3)*(u_22-u_11)*np.sin(2*a)*np.sin(2*b),
                   4*t_12*np.sin(a)*(np.cos(a)-np.cos(b))
                   +4*u_12*np.sin(2*a)*(np.cos(2*a)-np.cos(2*b))
                )
    V_22 = (e_2 + (3*t_11+t_22)*np.cos(a)*np.cos(b) + 2*t_22*np.cos(2*a)
            +2*r_11*(2*np.cos(3*a)*np.cos(b)+np.cos(2*b))
            +2/np.sqrt(3)*r_12*(4*np.cos(3*a)*np.cos(b)-np.cos(2*b))
            +(3*u_11+u_22)*np.cos(2*a)*np.cos(2*b) + 2*u_22*np.cos(4*a)
            )
    H_0 = np.array([[V_0,V_1,V_2],
                    [np.conjugate(V_1),V_11,V_12],
                    [np.conjugate(V_2),np.conjugate(V_12),V_22]])
    L_z = np.zeros((3,3),dtype = complex)
    L_z[1,2] = 2*1j;    L_z[2,1] = -2*1j
    Sig_3 = np.zeros((2,2),dtype=complex)
    Sig_3[0,0] = 1; Sig_3[1,1] = -1
    Hp = lamb/2*np.kron(Sig_3,L_z)
    Id = np.identity(2)
    H_final = np.kron(Id,H_0) + Hp
    return H_final

#####
#####Moire potential part
#####
#Here I construct the single Moire potential parts. Function of the reciprocal Moire vector -> g=0,..,5
#It is diagonal since it does not mix the bands and/or spins. 
#Different values for in-plane (indexes 1,2) and out-of-plane (index 0) orbits.
def V_g(g,params):          #g is a integer from 0 to 5
    V_G,psi_G,V_K,psi_K = params        #extract the parameters
    Id = np.zeros((6,6),dtype = complex)
    Id[0,0] = V_G*np.exp(1j*(-1)**(g%2)*psi_G);   Id[3,3] = Id[0,0]
    Id[1,1] = V_K*np.exp(1j*(-1)**(g%2)*psi_K);   Id[2,2] = Id[1,1]; Id[4,4] = Id[1,1]; Id[5,5] = Id[1,1]
    return Id

#####
#####Total Giga-Hamiltonian
#####
#Here I put all together to form the giga-Hamiltonian matrix by considering momentum space neighbors up to the cutoff N
#To do it I give a number to each of the mini BZs, starting from the central one and then one circle at a time, going anticlockwise.
#In the look-up table "lu" I put the positions of these mini-BZs, using coordinates in units of G1=(1,0), G2=(1/2,sqrt(3)/2). 
def total_H(K_,N,params_H,params_V,a_M):
    #Moiré reciprocal lattice vectors. I start from the first one and obtain the others by doing pi/3 rotations
    G = [4*np.pi/np.sqrt(3)/a_M*np.array([0,1])]    
    for i in range(1,6):
        G.append(np.tensordot(R_z(np.pi/3*i),G[0],1))
    #dimension of the Hamiltonian, which is 6 times the number of mini-BZs. 
    #For N=0 it is 1, for N=1 it is 6+1, for N=2 it is 12+6+1 ecc..
    n_cells = int(1+3*N*(N+1))*6
    H = np.zeros((n_cells,n_cells),dtype=complex)
    
    ############
    ############ Diagonal part and look-up table of site's coordinates
    ############
    #look up matrix of coordinates of single BZs, in units of G1=(1,0), G2=(1/2,sqrt(3)/2)
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
                new_lu = lu[-1]
                #Append the new mini-BZ by incrementing the previous with "m". 
                #Each value of m has to be repeated n times, which are counted by j. 
                lu.append((new_lu[0]+m[i][0],new_lu[1]+m[i][1]))
                if j == n-1:
                    i += 1
                    j = 0
                else:
                    j += 1
            #Take the correct momentum by adding the components of "lu" times the reciprocal 
            #lattice vectors to the initial momentum K_
            Kp = K_ + G[0]*lu[-1][0] + G[1]*lu[-1][1]
            #place the corresponding 6x6 Hamiltonian in its position
            H[s*6:s*6+6,s*6:s*6+6] = H_p(Kp,params_H)
    
    ############
    ############ Off diagonal part --> Moire potential. 
    ############ We are placing a Moiré 6x6 potential just in the off-diagonal parts connected by a 
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
                H[s*6:(s+1)*6,nn*6:(nn+1)*6] = V_g(g,params_V)
    return H

#Lorentzian
def lorentzian_weight(k,e,*pars):
    K2,E2,weight,K_,E_ = pars
    return abs(weight)/((k-K_)**2+K2)/((e-E_)**2+E2)

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








