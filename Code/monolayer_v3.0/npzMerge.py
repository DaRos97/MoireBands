"""
npzMerge.py
------------
Collect all .npz simulation result files for a given TMD material and merge
them into a single HDF5 file (.h5).

Only files whose name starts with  'res_<TMD>_'  are collected, where TMD is
supplied by the user (WSe2 or WS2).

Each .npz file must contain:
  - "elements"  : scalar float
  - "pars"      : 1-D numpy array of 43 floats

After the 'res_<TMD>_' prefix the filename encodes exactly 10 underscore-
separated floats:
  res_WSe2_<K1>_<K2>_<K3>_<K4>_<K5>_<K6>_<B1>_<B2>_<B3>_<B4>.npz
  └─ prefix ──┘ └────── Ks (6) ──────────┘ └───── Bs (4) ──────┘

Output HDF5 layout
------------------
  /elements (N,  6)  float64  – chi2 elements per run
  /pars     (N, 43)  float64  – pars array per run
  /Ks       (N,  6)  float64  – first 6 filename parameters
  /Bs       (N,  4)  float64  – last  4 filename parameters

Usage
-----
  python npzMergez.py --tmd WSe2
  python npzMerge.py --tmd WS2 --input /path/to/npzs
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

VALID_TMDS   = ("WSe2", "WS2")
N_KS         = 6
N_BS         = 4
N_PARAMS     = N_KS + N_BS   # 10 total
N_PARS_ARRAY = 43


def parse_filename_params(stem: str, prefix: str) -> tuple[list[float], list[float]]:
    """
    Strip the leading prefix from stem and parse the remaining 10
    underscore-separated floats into Ks (first 6) and Bs (last 4).

    Parameters
    ----------
    stem   : filename without extension, e.g. 'res_WSe2_0.1_2.5_...'
    prefix : expected prefix without trailing '_', e.g. 'res_WSe2'
    """
    # Remove prefix (includes the trailing '_' that separates it from params)
    param_str = stem[len(prefix) + 1:]   # +1 for the '_' separator
    parts = param_str.split("_")

    if len(parts) != N_PARAMS:
        raise ValueError(
            f"Expected {N_PARAMS} parameters after prefix, got {len(parts)}: {parts}"
        )

    values = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            raise ValueError(f"Token '{p}' is not a valid float.")

    return values[:N_KS], values[N_KS:]

def merge(input_dir: Path, tmd: str, output_path: Path):
    prefix  = f"res_{tmd}"
    pattern = "*.npz"
    all_files = sorted(input_dir.glob(pattern))

    # Filter to only files matching the expected prefix
    files = [f for f in all_files if f.stem.startswith(prefix + "_")]

    print(f"Found {len(all_files)} .npz file(s) in {input_dir}, "
          f"{len(files)} match prefix '{prefix}_'.")

    if not files:
        sys.exit(f"[ERROR] No files with prefix '{prefix}_' found.")

    elements, pars_list, Ks_list, Bs_list = [], [], [], []

    for f in files:
        # --- load npz ---
        try:
            data = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f"  [WARN] Skipping {f.name}: could not load ({e})")
            continue

        # --- check required fields ---
        missing = [field for field in ("elements", "pars") if field not in data]
        if missing:
            print(f"  [WARN] Skipping {f.name}: missing field(s) {missing}")
            continue

        elements_val = np.asarray(data["elements"], dtype=np.float64).ravel()
        pars_val     = np.asarray(data["pars"], dtype=np.float64).ravel()

        if pars_val.shape[0] != N_PARS_ARRAY:
            print(f"  [WARN] {f.name}: 'pars' has {pars_val.shape[0]} elements "
                  f"(expected {N_PARS_ARRAY}), keeping anyway")

        # --- parse filename parameters ---
        try:
            ks, bs = parse_filename_params(f.stem, prefix)
        except ValueError as e:
            print(f"  [WARN] Skipping {f.name}: {e}")
            continue

        elements.append(elements_val)
        pars_list.append(pars_val)
        Ks_list.append(ks)
        Bs_list.append(bs)

    n = len(elements)
    if n == 0:
        sys.exit("[ERROR] No valid files were loaded.")

    print(f"\nSuccessfully loaded {n} / {len(files)} file(s).")
    print(f"  Ks shape : ({n}, {N_KS})")
    print(f"  Bs shape : ({n}, {N_BS})")
    print(f"  pars shape : ({n}, {len(pars_list[0])})")

    # --- write HDF5 ---
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("elements",   data=np.vstack(elements).astype(np.float64), compression="gzip")
        h5.create_dataset("pars",       data=np.vstack(pars_list).astype(np.float64), compression="gzip")
        h5.create_dataset("Ks",         data=np.array(Ks_list,   dtype=np.float64), compression="gzip")
        h5.create_dataset("Bs",         data=np.array(Bs_list,   dtype=np.float64), compression="gzip")

        h5.attrs["tmd"]        = tmd
        h5.attrs["n_runs"]     = n
        h5.attrs["source_dir"] = str(input_dir.resolve())

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\nSaved → {output_path.resolve()}  ({size_mb:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(
        description="Merge .npz simulation results for a TMD material into a single HDF5 file."
    )
    parser.add_argument(
        "--tmd", "-t",
        required=True,
        choices=VALID_TMDS,
        help="TMD material (WSe2 or WS2). Only files named 'res_<TMD>_*.npz' are collected.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("./Data"),
        help="Directory containing the .npz files (default: ./Data/).",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        sys.exit(f"[ERROR] Input path is not a directory: {args.input}")

    output_path = Path(f"Data/merged_{args.tmd}.h5")
    merge(args.input, args.tmd, output_path)

if __name__ == "__main__":
    main()









