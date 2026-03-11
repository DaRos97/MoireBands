import numpy as np

p02_s3 = 98.23
p01_s3 = 181.87
px1_s3 = 186.06
px2_s3 = 361.49

x1_s3 = -0.1 + 0.1/(p01_s3-p02_s3)*(px1_s3-p01_s3)
x2_s3 = -0.1 + 0.1/(p01_s3-p02_s3)*(px2_s3-p01_s3)

d_s3 = x2_s3-x1_s3

p03_s11 = 88.14
p02_s11 = 149.24
px1_s11 = 180.31
px2_s11 = 357.4

x1_s11 = -0.2 + 0.1/(p02_s11-p03_s11)*(px1_s11-p02_s11)
x2_s11 = -0.2 + 0.1/(p02_s11-p03_s11)*(px2_s11-p02_s11)

d_s11 = x2_s11-x1_s11


print("d s11:%.5f \ns3:%.5f"%(d_s11,d_s3))

a1 = 3.32
a2 = 3.18

def aM(th):
    return 1/np.sqrt(1/a1**2+1/a2**2-2*np.cos(th)/a1/a2)

def th(k):
    return np.arccos(
        a1*a2/2*(
            1/a1**2 + 1/a2**2 - (3*k/4/np.pi)**2
        ))

k_s3 = d_s3/3
k_s11 = d_s11/3/2*np.sqrt(3)

th_s3 = th(k_s3)/np.pi*180
th_s11 = th(k_s11)/np.pi*180

print("Theta s3:%.3f"%th_s3)
print("Theta s11:%.3f"%th_s11)


