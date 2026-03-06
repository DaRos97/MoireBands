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
if 0:
    machine = cfs.get_machine(os.getcwd())
    n_chunks = 1
    chunk, listFn = utils.get_parameters(0,n_chunks=n_chunks)
    theta_deviation = 0      #Change here for \pm 0.3 degrees
    nShells = 2
    Vk,phiK = (0.006,106/180*np.pi)
    spreadE = 0.03      # in eV
    #listFn = '0.010000_0.040000_7_160_180_3_-1.600000_-1.700000_3_0.300000_0.400000_3'
    output_file = cfs.getFilename(
        ('full_edcGamma',theta_deviation,nShells,Vk,phiK,spreadE),
        dirname=utils.get_home_dn(machine)+"Data/",
        floatPrecision=3
    ) + listFn +  ".h5"
else:
    #output_file = 'Data/full_edcGamma_0_2_0.006_1.850_0.0300.000000_0.040000_21_160_180_11_-1.600000_-1.700000_21_0.300000_0.400000_21.h5'
    #output_file = 'Data/full_edcGamma_0_2_0.006_1.850_0.0300.000000_0.040000_21_120_180_13_-2.000000_2.000000_41_-1.000000_1.000000_21.h5'
    #output_file = 'Data/full_edcGamma_0_2_0.006_1.850_0.0300.002000_0.030000_15_150_180_16_-2.000000_2.000000_41_-2.000000_2.000000_41.h5'
    #output_file = 'Data/full_edcGamma_0_2_0.006_1.850_0.0300.007000_0.025000_19_160_180_21_-2.000000_-1.200000_41_0.700000_1.300000_31.h5'
    #output_file = 'Data/full_edcGamma_0_2_0.006_1.850_0.0300.007000_0.025000_19_0_358_180_-1.850000_-1.650000_41_0.950000_1.150000_41.h5'
    # New WS2
    output_file = 'Data/full_edcGamma_0_2_0.006_1.850_0.0300.007000_0.025000_19_160_180_21_-2.000000_0.000000_21_0.000000_2.000000_21.h5'

maxMeasure = 0.3

# Load HDF5
df = pd.read_hdf(output_file, key="results")

# Convert to NumPy array
dataFull = df.to_numpy()
dataFull[:,1] = dataFull[:,1]/np.pi*180
maskNan = ~np.isnan(dataFull[:,4])
data = dataFull[maskNan]

V_col   = 0
phi_col = 1
wp_col  = 2
wd_col  = 3
p1_col  = 4
p2_col  = 5
p3_col  = 6

""" Measures """

# measure of positions
positions = data[:, [p1_col, p2_col, p3_col]]
sigma_pos = np.std(positions, axis=0)
ARPES_positions = cfs.dic_params_edcG_positions[sample] - cfs.dic_params_offset[sample]
m_pos = np.sqrt(
    np.sum(((positions - ARPES_positions) / sigma_pos)**2, axis=1)
)
m_pos = np.sum(np.absolute(positions-ARPES_positions),axis=1)
data_pos = np.column_stack([data, m_pos])
mpos_col = data_pos.shape[1] - 1
maskPos = m_pos < maxMeasure
data_pos = data_pos[maskPos]

# measure of distances
distances = np.zeros((positions.shape[0],2))
distances[:,0] = positions[:,0] - positions[:,1]
distances[:,1] = positions[:,0] - positions[:,2]
sigma_dis = np.std(distances, axis=0)
ARPES_distances = np.array([ARPES_positions[0]-ARPES_positions[1],ARPES_positions[0]-ARPES_positions[2]])
m_dis = np.sqrt(
    np.sum(((distances - ARPES_distances) / sigma_dis)**2, axis=1)
)
m_dis = np.sum(np.absolute(distances-ARPES_distances),axis=1)
data_dis = np.column_stack([data, m_dis])
mdis_col = data_dis.shape[1] - 1
maskDis = m_dis < maxMeasure
data_dis = data_dis[maskDis]

# -------------------------------------------------
# MINIMUM MEASURE PROJECTOR
# -------------------------------------------------
""" Position V-phi """
phi_all_pos = np.unique(data_pos[:, phi_col])
V_all_pos   = np.unique(data_pos[:, V_col])
min_map_pos = np.full((len(V_all_pos), len(phi_all_pos)), np.nan)

for i, V in enumerate(V_all_pos):
    for j, phi in enumerate(phi_all_pos):
        mask = (
            (data_pos[:, V_col] == V) &
            (data_pos[:, phi_col] == phi)
        )
        vals_pos = data_pos[mask][:, mpos_col]
        if len(vals_pos) > 0:
            min_map_pos[i, j] = np.min(vals_pos)

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

""" Position wp-wd """
wp_all_pos = np.unique(data_pos[:, wp_col])
wd_all_pos   = np.unique(data_pos[:, wd_col])
min_map_pos_w = np.full((len(wd_all_pos), len(wp_all_pos)), np.nan)

for i, wd in enumerate(wd_all_pos):
    for j, wp in enumerate(wp_all_pos):
        mask = (
            (data_pos[:, wd_col] == wd) &
            (data_pos[:, wp_col] == wp)
        )
        vals_pos = data_pos[mask][:, mpos_col]
        if len(vals_pos) > 0:
            min_map_pos_w[i, j] = np.min(vals_pos)

""" Distance wp-wd """
wp_all_dis = np.unique(data_dis[:, wp_col])
wd_all_dis   = np.unique(data_dis[:, wd_col])
min_map_dis_w = np.full((len(wd_all_dis), len(wp_all_dis)), np.nan)

for i, wd in enumerate(wd_all_dis):
    for j, wp in enumerate(wp_all_dis):
        mask = (
            (data_dis[:, wd_col] == wd) &
            (data_dis[:, wp_col] == wp)
        )
        vals_dis = data_dis[mask][:, mdis_col]
        if len(vals_dis) > 0:
            min_map_dis_w[i, j] = np.min(vals_dis)

# -------------------------------------------------
# CREATE FIGURE
# -------------------------------------------------
fig = plt.figure(figsize=(13, 10))
s_ = 15
s_2 = 20

""" Figure V-phi positions """
ax = fig.add_subplot(221)

im = ax.imshow(
    min_map_pos,
    origin="lower",
    aspect="auto",
    extent=[phi_all_pos.min(), phi_all_pos.max(),
            V_all_pos.min(), V_all_pos.max()],
    cmap='plasma_r'
)

ax.set_xlabel(r"$\phi$",size=s_)
ax.set_ylabel(r"$V$ [eV]",size=s_)
ax.set_title("Positions Measure",size=s_2)
ax.set_ylim(np.min(dataFull[:,0]),np.max(dataFull[:,0]))
ax.set_xlim(np.min(dataFull[:,1]),np.max(dataFull[:,1]))

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"Minimum measure over $w_p$ and $w_d$",
               fontsize=s_)

""" Figure V-phi distance """
ax = fig.add_subplot(222)

im = ax.imshow(
    min_map_dis,
    origin="lower",
    aspect="auto",
    extent=[phi_all_dis.min(), phi_all_dis.max(),
            V_all_dis.min(), V_all_dis.max()],
    cmap='viridis_r'
)

ax.set_xlabel(r"$\phi$",size=s_)
ax.set_title("Distances Measure",size=s_2)
ax.set_ylim(np.min(dataFull[:,0]),np.max(dataFull[:,0]))
ax.set_xlim(np.min(dataFull[:,1]),np.max(dataFull[:,1]))

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"Minimum measure over $w_p$ and $w_d$",
               fontsize=s_)

""" Figure wp-wd positions """
ax = fig.add_subplot(223)

im = ax.imshow(
    min_map_pos_w,
    origin="lower",
    aspect="auto",
    extent=[wp_all_pos.min(), wp_all_pos.max(),
            wd_all_pos.min(), wd_all_pos.max()],
    cmap='plasma_r'
)

ax.set_xlabel(r"$w_p$ [eV]",size=s_)
ax.set_ylabel(r"$w_d$ [eV]",size=s_)
ax.set_ylim(np.min(dataFull[:,3]),np.max(dataFull[:,3]))
ax.set_xlim(np.min(dataFull[:,2]),np.max(dataFull[:,2]))

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"Minimum measure over $V$ and $\phi$",
               fontsize=s_)

""" Figure wp-wd distance """
ax = fig.add_subplot(224)

im = ax.imshow(
    min_map_dis_w,
    origin="lower",
    aspect="auto",
    extent=[wp_all_dis.min(), wp_all_dis.max(),
            wd_all_dis.min(), wd_all_dis.max()],
    cmap='viridis_r'
)

ax.set_xlabel(r"$w_p$ [eV]",size=s_)
ax.set_ylim(np.min(dataFull[:,3]),np.max(dataFull[:,3]))
ax.set_xlim(np.min(dataFull[:,2]),np.max(dataFull[:,2]))

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"Minimum measure over $V$ and $\phi$",
               fontsize=s_)

plt.subplots_adjust(
    left = 0.064,
    bottom = 0.057,
    right = 0.976,
    top = 0.96,
    wspace = 0.117,
    hspace = 0.14
)
plt.show()










































