import sys,os
cwd = os.getcwd()
if cwd[6:11] == 'dario':
    master_folder = cwd[:40]
elif cwd[:20] == '/home/users/r/rossid':
    master_folder = cwd[:20] + '/git/MoireBands/Code'
elif cwd[:13] == '/users/rossid':
    master_folder = cwd[:13] + '/git/MoireBands/Code'
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import glob
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt

maxMeasure = 0.017

if len(sys.argv)!=2:
    print("usage: python plot_edcK.py arg1\nWith: arg1 filename of set of data to plot.")
    exit()
fn = sys.argv[1]

""" Dirname and parameters load """
sample='S11'
df = pd.read_hdf(fn, key="results")

# Convert to NumPy array
data = df.to_numpy()
data[:,1] *= 180/np.pi
maskNan = ~np.isnan(data[:,2])
#data = dataFull[maskNan]

V_col   = 0
phi_col = 1
p1_col  = 2
p2_col  = 3

phi_all = np.unique(data[:, phi_col])
V_all   = np.unique(data[:, V_col])

""" Measures """
# measure of distances
positions = data[:, [p1_col, p2_col]]
ARPES_positions = cfs.dic_params_edcK_positions[sample] - cfs.dic_params_offset[sample]
distances = positions[:,0] - positions[:,1]
ARPES_distance = ARPES_positions[0]-ARPES_positions[1]

m_dis = np.absolute(distances-ARPES_distance)
data_dis = np.column_stack([data, m_dis])
maskDis = (m_dis < maxMeasure) & maskNan
data_dis = data_dis[maskDis]

min_dis = np.full((len(V_all), len(phi_all)), np.nan)
for i, V in enumerate(V_all):
    for j, phi in enumerate(phi_all):
        mask_pos = (
            (data_dis[:, V_col] == V) &
            (data_dis[:, phi_col] == phi)
        )
        vals_dis = data_dis[mask_pos][:, -1]
        if len(vals_dis) > 0:
            min_dis[i, j] = np.min(vals_dis)

""" Figure """
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot()
s_ = 15
s_2 = 20

gPhi,gV = np.meshgrid(phi_all,V_all)

im = ax.pcolormesh(
    gPhi,gV,
    min_dis,
    cmap='plasma_r',
    vmin=0,
    vmax=maxMeasure
)

ax.set_xlabel(r"$\phi$ [°]",size=s_)
ax.set_ylabel(r"$V$ [eV]",size=s_)
ax.tick_params(axis='both',labelsize=s_)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Distance measure from ARPES",size=s_)
cbar.ax.tick_params(labelsize=s_)

fig.tight_layout()
plt.show()










































