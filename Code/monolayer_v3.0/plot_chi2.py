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

import h5py
import numpy as np
cwd = os.getcwd()
master_folder = cwd[:40]
sys.path.insert(1, master_folder)
import utils

offsetK2 = 1e-5
offsetK3 = 1e-4

def load_h5(path: Path):
    """ Load data """
    with h5py.File(path, "r") as h5:
        elements = h5["elements"][:]    # (N,6)
        Ks       = h5["Ks"][:]          # (N, 6)
        Bs       = h5["Bs"][:]          # (N, 4)
        pars     = h5["pars"][:]        # (N,43)
    mask = (Ks[:,2] > offsetK3) & (Ks[:,1] > offsetK2)
    elements = elements[mask]
    Ks = Ks[mask]
    Bs = Bs[mask]
    pars = pars[mask]
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
    TMD = str(input_path).split('_')[-1][:-3]
    if not input_path.exists():
        sys.exit(f"[ERROR] File not found: {input_path}")

    elements, Ks, Bs, pars = load_h5(input_path)
    print(f"Loaded {elements.shape[0]} runs from {input_path}")

    chi2_b = elements[:,0]
    K2_M = elements[:,2]
    measure = chi2_b + K2_M
    global_idx = int(np.argmin(measure))

    utils.plot_measure(measure, Ks, Bs, global_idx, TMD, args.cutoff)

    if input("Plot best result? [y/N]")=='y':
        utils.plotResults(pars[global_idx],TMD,Ks[global_idx],Bs[global_idx],elements[global_idx])


if __name__ == "__main__":
    main()
