""" Plotting script for edc results for analysis purposes.
edc at G has 7 columns: Vg, phiG, wp, wd, p1, p2, p3
edc at K has 4 columns: Vk, phiK, p1, p2
"""
import sys,os
cwd = os.getcwd()
master_folder = cwd[:40]
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import glob
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
from pathlib import Path

""" Dirname and parameters load """
sample='S11'
if len(sys.argv)!=3:
    print("usage: python plot_edc.py arg1 arg2\n",
          "With: arg1 filename of set of data to plot, arg2 max value.")
    exit()
fn = sys.argv[1]
if not Path(fn).is_file():
    raise ValueError("Not fount file: ",fn)
BZpoint = fn.split('/')[-1][8]
if BZpoint not in {'K','G'}:
    raise ValueError("Not a good filename for the edc: ",fn)
maxMeasure = float(sys.argv[2])

""" Check on gap """
dn = fn[:-len(fn.split('/')[-1])]
newFn = fn.split('/')[-1][:8]+'Gap'+fn.split('/')[-1][8:]
gapFn = dn + newFn
gapData = False
if Path(gapFn).is_file():
    gapData = True
    print("Importing gaps")
    gaps = pd.read_hdf(gapFn, key="results").to_numpy()

""" Data columns """
data = pd.read_hdf(fn, key="results").to_numpy()
data[:,1] = data[:,1]/np.pi*180     # rad to deg

if BZpoint=='G':
    V_col   = 0
    phi_col = 1
    wp_col  = 2
    wd_col  = 3
    p1_col  = 4
    p2_col  = 5
    p3_col  = 6

    wp_all = np.unique(data[:, wp_col])
    wd_all   = np.unique(data[:, wd_col])
else:
    V_col   = 0
    phi_col = 1
    p1_col  = 2
    p2_col  = 3

phi_all = np.unique(data[:, phi_col])
V_all   = np.unique(data[:, V_col])

maskNan = ~np.isnan(data[:,p1_col])

""" Measures """
if BZpoint=='G':
    # measure of positions
    positions = data[:, [p1_col, p2_col, p3_col]]
    ARPES_positions = cfs.dic_params_edcG_positions[sample] - cfs.dic_params_offset[sample]
    m_pos = np.nansum(np.absolute(positions-ARPES_positions),axis=1)
    data_pos = np.column_stack([data, m_pos])
    mpos_col = data_pos.shape[1] - 1
    maskPos = (m_pos < maxMeasure) & maskNan
    data_pos = data_pos[maskPos]

    # measure of distances
    distances = np.zeros((positions.shape[0],2))
    distances[:,0] = positions[:,0] - positions[:,1]
    distances[:,1] = positions[:,0] - positions[:,2]
    ARPES_distances = np.array([ARPES_positions[0]-ARPES_positions[1],ARPES_positions[0]-ARPES_positions[2]])
    m_dis = np.nansum(np.absolute(distances-ARPES_distances),axis=1)
    data_dis = np.column_stack([data, m_dis])
    mdis_col = data_dis.shape[1] - 1
    maskDis = (m_dis < maxMeasure) & maskNan
    data_dis = data_dis[maskDis]

    Vmin = 0
    phimin = 0

    """ Minima """

    """ V-phi """
    min_pos = np.full((len(V_all), len(phi_all)), np.nan)
    min_dis = np.full((len(V_all), len(phi_all)), np.nan)
    for i, V in enumerate(V_all):
        for j, phi in enumerate(phi_all):
            mask_pos = (
                (data_pos[:, V_col] == V) &
                (data_pos[:, phi_col] == phi)
            )
            vals_pos = data_pos[mask_pos][:, mpos_col]
            if len(vals_pos) > 0:
                min_pos[i, j] = np.min(vals_pos)
            if abs(V-Vmin)<1e-7 and abs(phimin-phi)<1e-7:
                indmin = np.argmin(vals_pos)
                print(data_pos[mask_pos][indmin,wp_col])
                print(data_pos[mask_pos][indmin,wd_col])
            #
            mask_dis = (
                (data_dis[:, V_col] == V) &
                (data_dis[:, phi_col] == phi)
            )
            vals_dis = data_dis[mask_dis][:, mdis_col]
            if len(vals_dis) > 0:
                min_dis[i, j] = np.min(vals_dis)

    """ wp-wd """
    min_pos_w = np.full((len(wd_all), len(wp_all)), np.nan)
    min_dis_w = np.full((len(wd_all), len(wp_all)), np.nan)

    for i, wd in enumerate(wd_all):
        for j, wp in enumerate(wp_all):
            mask_pos = (
                (data_pos[:, wd_col] == wd) &
                (data_pos[:, wp_col] == wp)
            )
            vals_pos = data_pos[mask_pos][:, mpos_col]
            if len(vals_pos) > 0:
                min_pos_w[i, j] = np.min(vals_pos)
            mask_dis = (
                (data_dis[:, wd_col] == wd) &
                (data_dis[:, wp_col] == wp)
            )
            vals_dis = data_dis[mask_dis][:, mdis_col]
            if len(vals_dis) > 0:
                min_dis_w[i, j] = np.min(vals_dis)
else:
    # measure of distances
    positions = data[:, [p1_col, p2_col]]
    ARPES_positions = cfs.dic_params_edcK_positions[sample] - cfs.dic_params_offset[sample]
    distances = positions[:,0] - positions[:,1]
    ARPES_distance = ARPES_positions[0] - ARPES_positions[1]

    m_dis = np.absolute(distances-ARPES_distance)
    data_dis = np.column_stack([data, m_dis])
    maskDis = (m_dis < maxMeasure) & maskNan
    data_dis = data_dis[maskDis]
    gaps = gaps[maskNan]

    min_dis = np.full((len(V_all), len(phi_all)), np.nan)
    gap_grid = np.full((len(V_all), len(phi_all)), np.nan)
    for i, V in enumerate(V_all):
        for j, phi in enumerate(phi_all):
            mask_pos = (
                (data_dis[:, V_col] == V) &
                (data_dis[:, phi_col] == phi)
            )
            vals_dis = data_dis[mask_pos][:, -1]
            vals_gap = gaps[mask_pos][:, -1]
            if len(vals_dis) > 0:
                min_dis[i, j] = np.min(vals_dis)
                gap_grid[i, j] = vals_gap

""" Figure """
if BZpoint=='G':
    fig = plt.figure(figsize=(13, 10))
    s_ = 15
    s_2 = 20

    """ Figure V-phi positions """
    ax = fig.add_subplot(221)

    gPhi,gV = np.meshgrid(phi_all,V_all)
    im = ax.pcolormesh(
        gPhi,gV,
        min_pos,
        cmap='plasma_r',
    )

    ax.set_xlabel(r"$\phi$",size=s_)
    ax.set_ylabel(r"$V$ [eV]",size=s_)
    ax.set_ylim(0,0.020)
    ax.set_title("Positions Measure",size=s_2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Minimum measure over $w_p$ and $w_d$",
                   fontsize=s_)

    """ Figure V-phi distance """
    ax = fig.add_subplot(222)

    im = ax.pcolormesh(
        gPhi,gV,
        min_dis,
        cmap='viridis_r',
    )

    ax.set_xlabel(r"$\phi$",size=s_)
    ax.set_title("Distances Measure",size=s_2)
    ax.set_ylim(0,0.020)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Minimum measure over $w_p$ and $w_d$",
                   fontsize=s_)

    """ Figure wp-wd positions """
    ax = fig.add_subplot(223)

    gWp,gWd = np.meshgrid(wp_all,wd_all)
    im = ax.pcolormesh(
        gWp,gWd,
        min_pos_w,
        cmap='plasma_r',
    )

    ax.set_xlabel(r"$w_p$ [eV]",size=s_)
    ax.set_ylabel(r"$w_d$ [eV]",size=s_)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Minimum measure over $V$ and $\phi$",
                   fontsize=s_)

    """ Figure wp-wd distance """
    ax = fig.add_subplot(224)

    im = ax.pcolormesh(
        gWp,gWd,
        min_dis_w,
        cmap='viridis_r',
    )

    ax.set_xlabel(r"$w_p$ [eV]",size=s_)

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
else:
    s_ = 15
    s_2 = 20
    gPhi,gV = np.meshgrid(phi_all,V_all)

    if gapData:
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(121)
        #
        ax2 = fig.add_subplot(122)
        ax2.set_title("Gaps")
        im = ax2.pcolormesh(
            gPhi,gV,
            gap_grid,
            cmap='viridis_r',
            vmin=0,
            vmax=maxMeasure
        )

        ax2.set_xlabel(r"$\phi$ [°]",size=s_)
        ax2.set_ylabel(r"$V$ [eV]",size=s_)
        ax2.tick_params(axis='both',labelsize=s_)
        cbar = fig.colorbar(im, ax=ax2)
        cbar.set_label("Gap value",size=s_)
        cbar.ax2.tick_params(labelsize=s_)
    else:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot()

    im = ax1.pcolormesh(
        gPhi,gV,
        min_dis,
        cmap='plasma_r',
        vmin=0,
        vmax=maxMeasure
    )

    ax1.set_xlabel(r"$\phi$ [°]",size=s_)
    ax1.set_ylabel(r"$V$ [eV]",size=s_)
    ax1.tick_params(axis='both',labelsize=s_)
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("Distance measure from ARPES",size=s_)
    cbar.ax1.tick_params(labelsize=s_)

    fig.tight_layout()

plt.show()










































