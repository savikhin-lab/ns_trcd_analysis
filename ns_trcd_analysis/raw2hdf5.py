import numpy as np
import h5py
from .core import count_subdirs


def ingest(input_dir, output_file_path) -> None:
    num_shots = count_subdirs(input_dir)
    for shotdir in input_dir.iterdir():
        num_wl = count_subdirs(shotdir)
        break
    with h5py.File(output_file_path, "w") as outfile:
        outfile.create_dataset("data", (20_000, 3, num_shots, num_wl, 1))
        data = outfile["data"]
        # Axes:
        # 0 - Time
        # 1 - Channel
        # 2 - Shot number
        # 3 - Wavelength
        # 4 - Pump state
        wls = set()
        shot_index = 0
        for shotdir in input_dir.iterdir():
            if not shotdir.is_dir():
                continue
            wl_index = 0
            for wldir in shotdir.iterdir():
                if not wldir.is_dir():
                    continue
                wls.add(wldir.name)
                store_shot(data, wldir, shot_index, wl_index)
                wl_index += 1
            shot_index += 1
        wls = sorted([int(x) for x in wls])
        outfile.create_dataset("wavelengths", (num_wl,), data=wls)


def store_shot(arr, path, shot_idx, wl_idx) -> None:
    par = np.load(path / "par.npy")
    perp = np.load(path / "perp.npy")
    ref = np.load(path / "ref.npy")
    arr[:, 0, shot_idx, wl_idx, 0] = par
    arr[:, 1, shot_idx, wl_idx, 0] = perp
    arr[:, 2, shot_idx, wl_idx, 0] = ref
