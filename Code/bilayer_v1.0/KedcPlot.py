""" Here we plot the result of the kEDC -> no onterlayer so only V and phi.
"""
import numpy as np
import matplotlib.pyplot as plt

dn = "Data/kEDC/kedc_S11_2_2.800000/"
fn = dn + "vbest_-1.655000_0.325000_-3.141593_3.124139_360.npy"
listPhi = np.linspace(-np.pi,np.pi,360,endpoint=False)

data = np.load(fn)


fig = plt.figure()#figsize=(15,10))
ax = fig.add_subplot()

phiPoints = np.array(data[:,0],dtype=int)
ax.scatter(
    listPhi[phiPoints]/np.pi*180,
    data[:,1]*1e3,
    color='dodgerblue',
    marker='+'
)

ax.axvline(-106,color='r')

ax.set_title("EDC at K")
ax.set_xlabel("Phase (°)")
ax.set_ylabel("V [meV]")
plt.show()
