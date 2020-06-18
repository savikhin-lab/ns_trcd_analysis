import numpy as np
import h5py
from itertools import product
from typing import List
from .core import count_subdirs


def ingest(input_dir, output_file_path) -> None:
    num_shots = count_subdirs(input_dir)
    wls = collect_wavelengths(input_dir / "1")
    with h5py.File(output_file_path, "w") as outfile:
        # Axes:
        # 0 - Time
        # 1 - Channel
        # 2 - Shot number
        # 3 - Wavelength
        # 4 - Pump state
        outfile.create_dataset("data", (20_000, 3, num_shots, len(wls), 1))
        data = outfile["data"]
        outfile.create_dataset("wavelengths", (len(wls),), data=wls)
        dir_indices = product(range(1, num_shots+1), range(len(wls)))
        for shot_index, wl_index in dir_indices:
            print(f"Shot {shot_index}, Wavelength {wl_index}")
            datadir = input_dir / f"{shot_index}" / f"{wls[wl_index]}"
            # np.s_[...] generates the indices that you would normally get by slicing a NumPy array
            par = np.load(datadir / "par.npy")
            data.write_direct(par, np.s_[:], np.s_[:, 0, shot_index - 1, wl_index, 0])
            perp = np.load(datadir / "perp.npy")
            data.write_direct(perp, np.s_[:], np.s_[:, 1, shot_index - 1, wl_index, 0])
            ref = np.load(datadir / "ref.npy")
            data.write_direct(ref, np.s_[:], np.s_[:, 2, shot_index - 1, wl_index, 0])


def store_shot(arr, path, shot_idx, wl_idx) -> None:
    par = np.load(path / "par.npy")
    perp = np.load(path / "perp.npy")
    ref = np.load(path / "ref.npy")
    arr[:, 0, shot_idx, wl_idx, 0] = par
    arr[:, 1, shot_idx, wl_idx, 0] = perp
    arr[:, 2, shot_idx, wl_idx, 0] = ref


def collect_wavelengths(path) -> List[int]:
    """Collect the wavelengths from a shot directory.
    """
    wls = []
    for d in path.iterdir():
        if d.name[0] == "_":
            continue
        if not d.is_dir():
            continue
        wls.append(int(d.name))
    return wls
