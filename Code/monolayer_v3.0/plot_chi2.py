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
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_h5(path: Path):
    with h5py.File(path, "r") as h5:
        chi2 = h5["chi2"][:]   # (N,)
        Ks   = h5["Ks"][:]     # (N, 6)
        Bs   = h5["Bs"][:]     # (N, 4)
    return chi2, Ks, Bs


# ---------------------------------------------------------------------------
# Build 2D grid:  x = Ks[:,1],  y = Ks[:,2],  z = min(chi2) over the rest
# ---------------------------------------------------------------------------

def build_grid(chi2, Ks, Bs):
    x_vals = np.unique(Ks[:, 1])
    y_vals = np.unique(Ks[:, 2])

    grid = np.full((len(y_vals), len(x_vals)), np.nan)

    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    for i in range(len(chi2)):
        xi = x_idx[Ks[i, 1]]
        yi = y_idx[Ks[i, 2]]
        if np.isnan(grid[yi, xi]) or chi2[i] < grid[yi, xi]:
            grid[yi, xi] = chi2[i]

    return x_vals, y_vals, grid


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(chi2, Ks, Bs, tmd: str, output_path: Path | None):
    x_vals, y_vals, grid = build_grid(chi2, Ks, Bs)

    # --- global minimum ---
    global_idx = int(np.argmin(chi2))
    min_chi2   = chi2[global_idx]
    min_Ks     = Ks[global_idx]
    min_Bs     = Bs[global_idx]

    print(f"Global chi2 minimum : {min_chi2:.6g}  (index {global_idx})")
    print(f"  Ks = {min_Ks}")
    print(f"  Bs = {min_Bs}")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 6))

    img = ax.pcolormesh(
        x_vals, y_vals, grid,
        shading="nearest",
        cmap="viridis_r",
    )

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(r"$\min\,\chi^2$", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

    # Mark the global minimum on the grid
    ax.scatter(
        min_Ks[1], min_Ks[2],
        marker="*", s=220, color="red", zorder=5,
        label="global minimum",
    )

    # --- legend with full parameter values ---
    ks_str = "\n".join(f"  K{i} = {min_Ks[i]:.6g}" for i in range(len(min_Ks)))
    bs_str = "\n".join(f"  B{i} = {min_Bs[i]:.6g}" for i in range(len(min_Bs)))
    legend_text = (
        f"$\\chi^2_{{\\min}}$ = {min_chi2:.6g}\n"
        f"Ks:\n{ks_str}\n"
        f"Bs:\n{bs_str}"
    )

    ax.scatter([], [], marker="*", color="red", label=legend_text)
    ax.legend(
        fontsize=8,
        loc="upper right",
        framealpha=0.85,
        handlelength=1.2,
        borderpad=0.8,
    )

    ax.set_xlabel(r"$K_1$", fontsize=13)
    ax.set_ylabel(r"$K_2$", fontsize=13)
    ax.set_title(
        f"{tmd} — $\\chi^2$ map  "
        r"($\min$ over remaining parameters)",
        fontsize=13,
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved → {output_path.resolve()}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D chi2 map from merged HDF5 results."
    )
    parser.add_argument(
        "--tmd", "-t",
        required=True,
        choices=("WSe2", "WS2"),
        help="TMD material label (used for the default input filename and plot title).",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="HDF5 file to read (default: merged_<TMD>.h5).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save figure to this path instead of displaying it (e.g. chi2.png).",
    )
    args = parser.parse_args()

    input_path = args.input or Path(f"merged_{args.tmd}.h5")
    if not input_path.exists():
        sys.exit(f"[ERROR] File not found: {input_path}")

    chi2, Ks, Bs = load_h5(input_path)
    print(f"Loaded {len(chi2)} runs from {input_path}")

    plot(chi2, Ks, Bs, args.tmd, args.output)


if __name__ == "__main__":
    main()
