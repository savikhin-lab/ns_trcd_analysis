import h5py
import numpy as np
import json
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


def reject_fft(data, scale, upper, lower):
    """Reject shots that have too much noise in a given frequency band.
    """
    ffts = np.absolute(np.fft.rfft(data, axis=0))
    freqs = np.fft.fftfreq(data.shape[0], 0.02)[:(data.shape[0] // 2 + 1)]
    freqs[-1] *= -1  # Highest freq is always negative for whatever reason
    band_indices = (freqs > lower) & (freqs < upper)
    band_sums = np.apply_along_axis(lambda x: np.sum(x[band_indices]), 0, ffts)
    band_means = np.mean(band_sums, axis=0)
    rejected = {}
    for wl in range(data.shape[2]):
        rejected[wl] = []
        for shot in range(data.shape[1]):
            if band_sums[shot, wl] > scale * band_means[wl]:
                rejected[wl].append(shot)
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


def load_filter_list(filename):
    """Load a filter list from a JSON file.
    """
    with filename.open("r") as f:
        tmp = json.load(f)
    # Keys get loaded as strings, need to convert to ints
    loaded = {}
    for k in tmp.keys():
        loaded[int(k)] = tmp[k]
    return loaded


def merge_filter_lists(a, b):
    """Merge two lists of shots to filter.
    """
    merged_keys = set(a.keys())
    merged_keys.update(set(b.keys()))
    merged_keys = sorted(list(merged_keys))
    out = {}
    for i in merged_keys:
        x = a.get(i)
        y = b.get(i)
        if all([x, y]):
            merged = set(x)
            merged.update(set(y))
            merged = sorted(list(merged))
            out[i] = merged
        elif x is not None:
            out[i] = x
        else:
            out[i] = y
    return out
