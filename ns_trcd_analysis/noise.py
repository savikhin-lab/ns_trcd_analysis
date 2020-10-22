import click
import h5py
import numpy as np
from .core import POINTS


def reject_sigma(infile, sigmas):
    """Reject shots whose noise is a multiple of the average noise.
    """
    _, shots, num_wls = infile["data"].shape
    data = np.empty((POINTS, shots, num_wls))
    infile["data"].read_direct(data)
    rejected = dict()
    for wl_idx in range(num_wls):
        avg_stddev = data[:, :, wl_idx].std(axis=0).mean()
        rejected_shots = list()
        shot_noises = data[:, :, wl_idx].std(axis=0)
        for shot_idx in range(shots):
            if shot_noises[shot_idx] > (sigmas * avg_stddev):
                rejected_shots.append(shot_idx)
        rejected[wl_idx] = rejected_shots
    return rejected


def selective_average(infile, outfile, rejections):
    """Average dA or dCD by removing bad shots.
    """
    _, shots, num_wls = infile["data"].shape
    data = np.empty((POINTS, shots, num_wls))
    infile["data"].read_direct(data)
    for wl_idx in range(num_wls):
        for shot_idx in rejections[wl_idx]:
            data[:, shot_idx, wl_idx] = 0
    average = data.mean(axis=1)
    for wl_idx in range(num_wls):
        scale_factor = shots / (shots - len(rejections[wl_idx]))
        average[:, wl_idx] *= scale_factor
    with h5py.File(outfile, "w") as outfile:
        outfile.copy(infile["wavelengths"], "wavelengths")
        outfile.create_dataset("data", data=data)
        outfile.create_dataset("average", data=average)
    return