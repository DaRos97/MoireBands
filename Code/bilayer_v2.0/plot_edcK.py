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

""" Dirname and parameters load """
sample='S11'

output_file = 'Data/full_edcK_0_2_0.019_3.054_-1.750_1.050_0.030_0.001000_0.070000_70_0_359_360.h5'

maxMeasure = 0.1

# Load HDF5
df = pd.read_hdf(output_file, key="results")

# Convert to NumPy array
dataFull = df.to_numpy()
dataFull[:,1] = dataFull[:,1]/np.pi*180
maskNan = ~np.isnan(dataFull[:,2])
data = dataFull[maskNan]

V_col   = 0
phi_col = 1
p1_col  = 2
p2_col  = 3

""" Measures """
# measure of distances
positions = data[:, [p1_col, p2_col]]
ARPES_positions = cfs.dic_params_edcK_positions[sample] - cfs.dic_params_offset[sample]
distances = positions[:,0] - positions[:,1]
ARPES_distance = ARPES_positions[0]-ARPES_positions[1]
m_dis = np.absolute(distances-ARPES_distance)

data_dis = np.column_stack([data, m_dis])
mdis_col = data_dis.shape[1] - 1
maskDis = m_dis < maxMeasure
data_dis = data_dis[maskDis]

# -------------------------------------------------
# MINIMUM MEASURE PROJECTOR
# -------------------------------------------------
""" Distance V-phi """
phi_all_dis = np.unique(data_dis[:, phi_col])
V_all_dis   = np.unique(data_dis[:, V_col])
min_map_dis = np.full((len(V_all_dis), len(phi_all_dis)), np.nan)

for i, V in enumerate(V_all_dis):
    for j, phi in enumerate(phi_all_dis):
        mask = (
            (data_dis[:, V_col] == V) &
            (data_dis[:, phi_col] == phi)
        )
        vals_dis = data_dis[mask][:, mdis_col]
        if len(vals_dis) > 0:
            min_map_dis[i, j] = np.min(vals_dis)

""" Figure """
fig = plt.figure(figsize=(10, 10))
s_ = 15
s_2 = 20

""" Figure V-phi distance """
ax = fig.add_subplot()

im = ax.imshow(
    min_map_dis,
    origin="lower",
    aspect="auto",
    extent=[phi_all_dis.min(), phi_all_dis.max(),
            V_all_dis.min(), V_all_dis.max()],
    cmap='viridis_r'
)

ax.set_xlabel(r"$\phi$",size=s_)
ax.set_ylabel(r"$V$ [eV]",size=s_)
ax.set_title("Distances Measure",size=s_2)
ax.set_ylim(np.min(dataFull[:,0]),np.max(dataFull[:,0]))
ax.set_xlim(np.min(dataFull[:,1]),np.max(dataFull[:,1]))

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"Minimum measure over $w_p$ and $w_d$",
               fontsize=s_)

fig.tight_layout()
plt.show()










































