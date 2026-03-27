"""
plot_chi2.py
------------
Load a merged HDF5 results file and plot a 2D map of chi2 with:
  - x axis : Ks[:, 1]   (second K parameter)
  - y axis : Ks[:, 2]   (third  K parameter)
  - color  : minimum chi2 over all other fields (Ks[:,0], Ks[:,3:], Bs)

A legend box shows the full Ks and Bs values of the global chi2 minimum.

Usage
-----
  python plot_chi2.py --tmd WSe2
  python plot_chi2.py --tmd WS2  --input merged_WS2.h5
  python plot_chi2.py --tmd WSe2 --input merged_WSe2.h5 --output chi2_map.png
"""

import argparse
import sys,os
from pathlib import Path
import matplotlib.pyplot as plt

import h5py
import numpy as np
cwd = os.getcwd()
master_folder = cwd[:40]
sys.path.insert(1, master_folder)
import utils

offsetK1 = -1e-7
K2min = -2**(-8)
K2max = 10
offsetK3 = -0.012
offsetK6 = -1

def load_h5(path: Path,TMD: str):
    """ Load data """
    with h5py.File(path, "r") as h5:
        elements = h5["elements"][:]    # (N,6)
        Ks       = h5["Ks"][:]          # (N, 6)
        Bs       = h5["Bs"][:]          # (N, 4)
        pars     = h5["pars"][:]        # (N,43)
    # Pars value mask
    mask = (
        (Ks[:,0] > offsetK1) &
        (Ks[:,1] > K2min) &
        (Ks[:,1] < K2max) &
        (Ks[:,2] > offsetK3) &
        (Ks[:,5] > offsetK6)
    )
    elements = elements[mask]
    Ks = Ks[mask]
    Bs = Bs[mask]
    pars = pars[mask]
    # Pars bounds mask
    if TMD=='WSe2':
        tol = 1e-2
        mask2 = (
            (np.abs(pars[:, 0:7]) < Bs[:, [0]] - tol).all(axis=1) &
            (np.abs(pars[:, 7:28]) < Bs[:, [1]] - tol).all(axis=1) &
            (np.abs(pars[:, 28:36]) < Bs[:, [2]] - tol).all(axis=1) &
            (np.abs(pars[:, 36:40]) < Bs[:, [3]] - tol).all(axis=1)
        )
        elements = elements[mask2]
        Ks = Ks[mask2]
        Bs = Bs[mask2]
        pars = pars[mask2]
    return elements, Ks, Bs, pars

def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D chi2 map from merged HDF5 results."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="HDF5 file to read (e.g. Data/merged_WSe2.h5).",
    )
    parser.add_argument(
        "--cutoff", "-c",
        type=float,
        default=0.3,
        help="Cutoff value of chi2 (default: 0.3).",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        sys.exit(f"[ERROR] File not found: {input_path}")
    TMD = str(input_path).split('_')[1]
    if TMD not in ['WSe2','WS2']:
        sys.exit(f"[ERROR] File dot in right format: {input_path}")
    boundType = str(input_path).split('_')[-1][:-3]

    elements, Ks, Bs, pars = load_h5(input_path,TMD)
    print(f"Loaded {elements.shape[0]} runs from {input_path}")

    chi2 = elements[:,0]
    K2_M = elements[:,2]

    indChosen = 1 if TMD =='WSe2' else 0

    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(121)
    global_idx1 = np.argsort(chi2)[indChosen]
    utils.plot_measure(chi2, Ks, Bs, global_idx1, TMD, args.cutoff, 'chi2', fig, ax1)
    print('-'*20)
    ax2 = fig.add_subplot(122)
    global_idx2 = np.argsort(chi2+K2_M)[indChosen]
    utils.plot_measure(chi2+K2_M, Ks, Bs, global_idx2, TMD, args.cutoff, 'chi2+K2_M', fig, ax2)

    plt.tight_layout()
    plt.show()

    inp = input("Plot best result? [1-2-a/N] (a is for all): ")
    #inp = 'a'

    global_idxs = []
    if inp == '1':
        global_idxs.append(global_idx1)
    elif inp == '2':
        global_idxs.append(global_idx2)
    elif inp=='a':
        global_idxs = [global_idx1,global_idx2]
    for global_idx in global_idxs:
        utils.plotResults(pars[global_idx],TMD,Ks[global_idx],boundType,Bs[global_idx],elements[global_idx])

    if inp in ['1','2']:
        save = input("Save result? [y/N] ")=='y'
        if save:
            print("Saving result "+inp+" in file Data/result_"+TMD+".npy")
            parsResult = pars[global_idxs[0]]
            np.save("Data/result_"+TMD+".npy",parsResult)

if __name__ == "__main__":
    main()
