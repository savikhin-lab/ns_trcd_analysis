import numpy as np
import h5py
from .core import count_subdirs


def ingest(input_dir, output_file_path) -> None:
    num_shots = count_subdirs(input_dir)
    # Reserve space for all the measurements:
    # Axes:
    # 0 - Time
    # 1 - Channel
    # 2 - Shot number
    # 3 - Wavelength
    # 4 - Pump state
    data = np.empty((20_000, 3, num_shots, 1, 1))
    shot_index = 0
    for d in input_dir.iterdir():
        if not d.is_dir():
            continue
        store_shot(data, d, shot_index)
        shot_index += 1
    with h5py.File(output_file_path, "w") as file:
        file.create_dataset("data", data.shape, data=data)


def store_shot(arr, path, idx) -> None:
    par = np.load(path / "par.npy")
    perp = np.load(path / "perp.npy")
    ref = np.load(path / "ref.npy")
    arr[:, 0, idx, 0, 0] = par
    arr[:, 1, idx, 0, 0] = perp
    arr[:, 2, idx, 0, 0] = ref
