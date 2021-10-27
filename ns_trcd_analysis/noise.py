import click
import h5py
import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from .compute import multi_exp
from .core import POINTS, time_axis


def reject_sigma(data, sigmas):
    """Reject shots whose noise is a multiple of the average noise.
    """
    _, shots, num_wls = data.shape
    rejected = {}
    with click.progressbar(range(num_wls), label="Rejecting by sigma") as wl_indices:
        for wl_idx in wl_indices:
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
    t_points, shots, num_wls = data.shape
    ffts = np.absolute(np.fft.rfft(data, axis=0))
    freqs = np.fft.fftfreq(t_points, 0.02)[:(t_points // 2 + 1)]
    freqs[-1] *= -1  # Highest freq is always negative for whatever reason
    band_indices = (freqs > lower) & (freqs < upper)
    band_sums = np.apply_along_axis(lambda x: np.sum(x[band_indices]), 0, ffts)
    band_means = np.mean(band_sums, axis=0)
    rejected = {}
    with click.progressbar(range(num_wls), label="Rejecting by FFT") as wl_indices:
        for wl in wl_indices:
            rejected[wl] = []
            for shot in range(shots):
                if band_sums[shot, wl] > scale * band_means[wl]:
                    rejected[wl].append(shot)
    return rejected


def reject_integral(data, scale, start, stop):
    """Reject shots based on the integral between a start and stop time.
    """
    _, shots, num_wls = data.shape
    ts = time_axis()
    t_range = (ts > start) & (ts < stop)
    sums = np.absolute(np.sum(data[t_range, :, :], axis=0))
    means = np.mean(sums, axis=0)
    rejected = {}
    with click.progressbar(range(num_wls), label="Rejecting by integral") as wl_indices:
        for wl in wl_indices:
            rejected[wl] = []
            for shot in range(shots):
                if sums[shot, wl] < scale * means[wl]:
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
    with click.progressbar(range(num_wls), label="Averaging") as wl_indices:
        for wl_idx in wl_indices:
            scale_factor = shots / (shots - len(rejections[wl_idx]))
            average[:, wl_idx] *= scale_factor
    with h5py.File(outfile, "w") as outfile:
        outfile.copy(infile["wavelengths"], "wavelengths")
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
    merged_keys.union(set(b.keys()))
    merged_keys = sorted(list(merged_keys))
    out = {}
    for i in merged_keys:
        x = set(a.get(i))
        y = set(b.get(i))
        merged = x.union(y)
        out[i] = sorted(list(merged))
    return out


def filter_from_fits(da, fits, collapsed_t, scale):
    """Use the global fits to determine the noise in individual curves for rejection.
    """
    points, shots, n_wls = da.shape
    rejections = dict()
    # The time axis is often shifted during processing (by just a point or two),
    # so we want to interpolate with the shifted time axis rather than the default
    # time axis.
    shifted_t = time_axis()
    shifted_t += collapsed_t[0] - shifted_t[0]
    # The collapsed time doesn't extend all the way to the end of the time interval
    # (since those last few points were collapsed), so we can't interpolate all the
    # way to the end of the uncollapsed time axis. We need to cut the last few time
    # points from the generated time axis.
    limited_t = shifted_t[shifted_t < collapsed_t[-1]]
    last_t = shifted_t[shifted_t < collapsed_t[-1]].shape[0]
    for wl_index in range(n_wls):
        interpolator = interp1d(collapsed_t, fits[:, wl_index])
        da_interp_1d = interpolator(limited_t)
        da_interp = np.empty_like(da[:last_t, :, wl_index])
        for i in range(n_wls):
            da_interp[:, i] = da_interp_1d
        stds = np.std(da[:last_t, :, wl_index] - da_interp, axis=0)
        stds_std = stds.std()
        stds_mean = stds.mean()
        wl_rejections = list()
        for shot_index in range(shots):
            if stds[shot_index] > (scale * stds_std + stds_mean):
                wl_rejections.append(shot_index)
        rejections[wl_index] = wl_rejections
    return rejections


def incremental_filter(da, filtered, threshold):
    """Build a list of rejected shots by examining how the noise changes
    when the shot is eliminated.
    """
    _, shots, n_wls = da.shape
    fit_after = 0.25
    # 4 amplitudes followed by 4 lifetimes
    guesses = [-1e-3, -1e-3, -1e-3, -1e-3, 0.25, 1.5, 30, 55]
    bounds = ([-0.02, -0.02, -0.02, -0.02, 0.1, 1, 25, 45], [0.02, 0.02, 0.02, 0.02, 0.5, 4, 35, 70])
    ts = time_axis()
    da = da[ts > fit_after, :, :]
    ts = ts[ts > fit_after]
    with click.progressbar(range(n_wls-1), label="Filtering incrementally") as wl_indices:
        for wl_index in wl_indices:
            previous_noise = 10
            current_noise = 1
            try:
                keep = {x for x in range(shots)} - set(filtered[wl_index])
                keep = {x: 0 for x in keep}
            except KeyError:
                keep = {x: 0 for x in range(shots)}
            relative_noise = 1
            avg = avg_specified_shots(da, wl_index, keep.keys())
            while relative_noise > threshold:
                avg = avg_specified_shots(da, wl_index, keep.keys())
                fit_params, _ = curve_fit(multi_exp, ts, avg, p0=guesses, bounds=bounds)
                fitted = multi_exp(ts, *fit_params)
                previous_noise = current_noise
                current_noise = np.std(fitted - avg)
                relative_noise = abs(previous_noise - current_noise) / previous_noise
                for k in keep.keys():
                    residual_noise = np.std(da[:, k, wl_index] - avg)
                    keep[k] = residual_noise
                ordered_by_noise = [k for k, v in sorted(keep.items(), key=lambda x: x[1])]
                del keep[ordered_by_noise[-1]]
            rejects = set(filtered[wl_index]).union({x for x in range(shots)} - set(keep.keys()))
            filtered[wl_index] = sorted(list(rejects))
    return filtered


def avg_specified_shots(da, wl_idx, keep):
    to_avg = np.empty((da.shape[0], len(keep)))
    for i, shot_idx in enumerate(keep):
        to_avg[:, i] = da[:, shot_idx, wl_idx]
    avg = np.nanmean(to_avg, axis=1)
    return avg
