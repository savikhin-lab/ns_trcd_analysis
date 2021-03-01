# The shape of the input dataset is as follows:
# (points, channels, shots, wavelengths, pump states)
#
# The shape of the output dataset is as follows:
# (points, shots, wavelengths)
#
import click
import numpy as np
from itertools import product
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit
from . import core


POINTS_BEFORE_PUMP = 1_500
FIT_START_POINT = 1921


def compute_da_always_pumped(infile, outfile):
    """Compute dA from the raw parallel and reference channels when every shot has pump.
    """
    raw_ds = infile["data"]
    points, _, shots, wavelengths, _ = raw_ds.shape
    tmp_raw = np.empty((points, 3, shots, wavelengths, 1))
    tmp_da = np.empty((points, shots, wavelengths))
    raw_ds.read_direct(tmp_raw)
    with click.progressbar(range(shots), label="Computing dA") as shots:
        for shot_idx in shots:
            for wl_idx in range(wavelengths):
                par = tmp_raw[:, 0, shot_idx, wl_idx, 0]
                ref = tmp_raw[:, 2, shot_idx, wl_idx, 0]
                before_zero_par = par[:POINTS_BEFORE_PUMP]
                before_zero_ref = ref[:POINTS_BEFORE_PUMP]
                without_pump = np.mean(before_zero_par / before_zero_ref)
                tmp_da[:, shot_idx, wl_idx] = -np.log10(par / ref / without_pump)
    outfile["data"].write_direct(tmp_da)
    return


def compute_da_with_and_without_pump(infile, outfile):
    """Compute dA from the raw parallel and reference channels with and without pump.
    """
    raw_ds = infile["data"]
    points, _, shots, wavelengths, _ = raw_ds.shape
    tmp_raw = np.empty((points, 3, shots, wavelengths, 2))
    tmp_da = np.empty((points, shots, wavelengths))
    raw_ds.read_direct(tmp_raw)
    with click.progressbar(range(shots), label="Computing dA") as shots:
        for shot_idx in shots:
            for wl_idx in range(wavelengths):
                par_np = tmp_raw[:, 0, shot_idx, wl_idx, 1]
                ref_np = tmp_raw[:, 2, shot_idx, wl_idx, 1]
                par_wp = tmp_raw[:, 0, shot_idx, wl_idx, 0]
                ref_wp = tmp_raw[:, 2, shot_idx, wl_idx, 0]
                tmp_da[:, shot_idx, wl_idx] = -np.log10((par_wp / ref_wp) / (par_np / ref_np))
    outfile["data"].write_direct(tmp_da)
    return


def compute_perp_da(infile, outfile):
    """Compute dA from the raw perpendicular and reference channels.
    """
    raw_ds = infile["data"]
    points, _, shots, wavelengths, _ = raw_ds.shape
    tmp_raw = np.empty((points, 3, shots, wavelengths, 1))
    tmp_da = np.empty((points, shots, wavelengths))
    raw_ds.read_direct(tmp_raw)
    with click.progressbar(range(shots), label="Computing dA") as shots:
        for shot_idx in shots:
            for wl_idx in range(wavelengths):
                perp = tmp_raw[:, 1, shot_idx, wl_idx, 0]
                ref = tmp_raw[:, 2, shot_idx, wl_idx, 0]
                before_zero_perp = perp[:POINTS_BEFORE_PUMP]
                before_zero_ref = ref[:POINTS_BEFORE_PUMP]
                without_pump = np.mean(before_zero_perp / before_zero_ref)
                tmp_da[:, shot_idx, wl_idx] = -np.log10(perp / ref / without_pump)
    outfile["data"].write_direct(tmp_da)
    return


def compute_cd_always_pumped(infile, outfile, delta):
    """Compute dCD from the raw parallel and perpendicular channels when every shot is pumped.
    """
    ds_in = infile["data"]
    points, _, shots, wavelengths, _ = ds_in.shape
    tmp_raw = np.empty((points, 3, shots, wavelengths, 1))
    tmp_cd = np.empty((points, shots, wavelengths))
    ds_in.read_direct(tmp_raw)
    coeff = (4 * delta) / 2.3
    with click.progressbar(range(shots), label="Computing CD") as shots:
        for shot_idx in shots:
            for wl_idx in range(wavelengths):
                par = tmp_raw[:, 0, shot_idx, wl_idx, 0]
                perp = tmp_raw[:, 1, shot_idx, wl_idx, 0]
                before_zero_par = par[:POINTS_BEFORE_PUMP]
                before_zero_perp = perp[:POINTS_BEFORE_PUMP]
                without_pump = np.mean(before_zero_perp / before_zero_par)
                tmp_cd[:, shot_idx, wl_idx] = coeff * (perp / par - without_pump)
    outfile["data"].write_direct(tmp_cd)
    return


def compute_cd_with_and_without_pump(infile, outfile, delta):
    """Compute dCD from the raw parallel and perpendicular channels with and without pump.
    """
    ds_in = infile["data"]
    points, _, shots, wavelengths, _ = ds_in.shape
    tmp_raw = np.empty((points, 3, shots, wavelengths, 2))
    tmp_cd = np.empty((points, shots, wavelengths))
    ds_in.read_direct(tmp_raw)
    coeff = (4 * delta) / 2.3
    with click.progressbar(range(shots), label="Computing CD") as shots:
        for shot_idx in shots:
            for wl_idx in range(wavelengths):
                par_np = tmp_raw[:, 0, shot_idx, wl_idx, 1]
                perp_np = tmp_raw[:, 1, shot_idx, wl_idx, 1]
                par_wp = tmp_raw[:, 0, shot_idx, wl_idx, 0]
                perp_wp = tmp_raw[:, 1, shot_idx, wl_idx, 0]
                tmp_cd[:, shot_idx, wl_idx] = coeff * (perp_wp / par_wp - perp_np / par_np)
    outfile["data"].write_direct(tmp_cd)
    return


def average(f):
    """Average all measurements for each wavelength.
    """
    da_ds = f["data"]
    points, shots, wls = da_ds.shape
    avg_ds = f.create_dataset("average", (points, wls))
    avg_ds.write_direct(np.nanmean(da_ds, axis=1))
    return


def subtract_background(f) -> None:
    """Subtract a linear background from a set of dA curves.
    """
    da_ds = f["data"]
    points, shots, wls = da_ds.shape
    x = np.arange(points)
    t_before_pump = x[:POINTS_BEFORE_PUMP]
    tmp_da = np.empty((points, shots, wls))
    da_ds.read_direct(tmp_da)
    meas_indices = [x for x in product(range(shots), range(wls))]
    with click.progressbar(meas_indices, label="Subtracting background") as indices:
        for shot_idx, wl_idx in indices:
            da_before_pump = tmp_da[:POINTS_BEFORE_PUMP, shot_idx, wl_idx]
            (slope, intercept), _ = curve_fit(line, t_before_pump, da_before_pump)
            background = line(x, slope, intercept)
            tmp_da[:, shot_idx, wl_idx] -= background
    da_ds.write_direct(tmp_da)
    return


def line(x, m, b) -> np.ndarray:
    """Compute a line for use with background subtraction.
    """
    return m * x + b


def remove_da_shot_offsets(dataset, offset_points):
    """Remove the before-pump offset from each shot.
    """
    points, shots, wls = dataset.shape
    total_shots = shots * wls
    all_indices = [x for x in product(range(shots), range(wls))]
    original = np.empty((points, shots, wls))
    tmp = np.empty_like(original)
    dataset.read_direct(original)
    with click.progressbar(all_indices, label="Subtracting offsets") as indices:
        for shot_idx, wl_idx in indices:
            shot = original[:, shot_idx, wl_idx]
            before_pump_avg = shot[:offset_points].mean()
            shot -= before_pump_avg
            tmp[:, shot_idx, wl_idx] = shot
    dataset.write_direct(tmp)
    return


def remove_avg_offsets(dataset, offset_points, ds_name="average"):
    """Remove the before-pump offset from dA or dCD averages.
    """
    points, wls = dataset.shape
    original = np.empty((points, wls))
    tmp = np.empty_like(original)
    dataset.read_direct(original)
    with click.progressbar(range(wls), label="Subtracting offsets") as indices:
        for wl_idx in indices:
            shot = original[:, wl_idx]
            before_pump_avg = shot[:offset_points].mean()
            shot -= before_pump_avg
            tmp[:, wl_idx] = shot
    dataset.write_direct(tmp)
    return


def collapse(data, times, cpoints):
    """Reduce the number of points in a dataset by averaging multiple points together.
    """
    # Just copy the whole array, we'll need all of the points before the second period anyway
    tmp = np.copy(data)
    # Fast forward to the beginning of the second period
    orig_t_idx = 0
    cutoff_indices = []
    for t in times:
        while True:
            if data[orig_t_idx, 0] < t:
                orig_t_idx += 1
            else:
                cutoff_indices.append(orig_t_idx)
                break
    cutoff_indices.append(data.shape[0]+1)
    output_idx = cutoff_indices[0]
    for i in range(len(cutoff_indices)-1):
        start = cutoff_indices[i]
        stop = cutoff_indices[i+1]
        num_splits = np.ceil((stop - start)/cpoints[i])
        splits = np.array_split(data[start:stop, :], num_splits)
        for s in splits:
            tmp[output_idx, :] = s.mean(axis=0)
            output_idx += 1
    output_data = tmp[:output_idx, :]
    return output_data


def multi_exp(x, *args) -> np.ndarray:
    """Compute a multi-exponential decay function.

    The first argument is the time axis. The arguments that follow must be in the
    order 'a1', 'a2', ..., 'an', 't1', 't2', ..., 'tn'.
    """
    out = np.zeros_like(x)
    arg_list = list(args)
    amplitudes = arg_list[:(len(arg_list)//2)]
    lifetimes = arg_list[(len(arg_list)//2):]
    for a, tau in zip(amplitudes, lifetimes):
        this_exp = a * np.exp(-x / tau)
        np.add(out, this_exp, out=out)
    return out


def make_lfit_param_list(amps, ls):
    """Interleave the lifetimes and parameters to pass them to the multi_exp function.
    """
    return tuple(amps + ls)


def lfits_for_gfit(data, ts, fit_after_time, lifetimes, bounds):
    """Do local fits for each curve to provide a starting point for the global fit.
    """
    lfit_params = np.empty((len(lifetimes), data.shape[1]))
    lower = []
    upper = []
    guesses = []
    # Bounds for the amplitude
    for i in range(len(lifetimes)):
        lower.append(bounds[i][0])
        upper.append(bounds[i][1])
    # Bounds for the lifetime
    for i in range(len(lifetimes)):
        lower.append(0.5 * lifetimes[i])
        upper.append(1.5 * lifetimes[i])
    amp_guesses = [-0.001 for x in range(len(lifetimes))]
    guesses = amp_guesses + lifetimes
    fit_bounds = (lower, upper)
    for i in range(data.shape[1]):
        ys = data[:, i]
        res, _ = curve_fit(multi_exp, ts[ts > fit_after_time], ys[ts > fit_after_time], p0=guesses, bounds=fit_bounds)
        amplitudes = res[:len(lifetimes)]
        lfit_params[:, i] = np.asarray(amplitudes)
    return lfit_params


def make_gfit_bounds(num_wls, lifetimes, bounds):
    """Make the bounds lists for the global fit.

    There are two lists: the upper bounds and lower bounds. If there are N lifetimes
    and M wavelengths, each bounds list will have this format:
        a11, a12, ..., a1N, ..., aMN, t1, t2, ..., tN
    """
    single_wavelength_lower = [bounds[i][0] for i in range(len(bounds))]
    single_wavelength_upper = [bounds[i][1] for i in range(len(bounds))]
    lifetimes_lower = [0.5 * l for l in lifetimes]
    lifetimes_upper = [1.5 * l for l in lifetimes]
    lower = num_wls * single_wavelength_lower + lifetimes_lower
    upper = num_wls * single_wavelength_upper + lifetimes_upper
    return (lower, upper)


def make_gfit_guesses(lfits, lifetimes):
    """Make the initial guesses for the global fit.
    
    If there are N lifetimes and M wavelengths, this list has the format:
        a11, a12, ..., a1N, ..., aMN, t1, t2, ..., tN
    """
    guesses = []
    for i in range(lfits.shape[1]):
        this_fit = list(lfits[:, i])
        guesses += this_fit
    guesses += lifetimes
    return guesses


def gfit_amp_arr_from_args(params, n_lifetimes, n_wls):
    """Reshape the parameter list into an array for easy indexing.
    """
    p_arr = np.empty((n_lifetimes, n_wls))
    for i in range(n_lifetimes):
        for j in range(n_wls):
            list_index = i * n_wls + j
            p_arr[i, j] = params[list_index]
    return p_arr


def global_fit(data, ts, fit_after, lfits, lifetimes, bounds):
    """Do a global fit of the data using local fits as the starting point.
    """
    n_lifetimes = len(lifetimes)
    n_wls = data.shape[1]
    gfit_bounds = make_gfit_bounds(lfits.shape[1], lifetimes, bounds)
    gfit_guesses = make_gfit_guesses(lfits, lifetimes)
    data_for_fit = data[ts > fit_after, :]
    ts_for_fit = ts[ts > fit_after]
    # Flatten the arrays into a single column since the curve_fit function
    # requires that the x- and y-array be 1D.
    xs = np.repeat(ts_for_fit, n_wls)
    ys = data_for_fit.reshape(data_for_fit.shape[0] * data_for_fit.shape[1])

    def fit_me(x, *args):
        fitted = np.empty_like(data_for_fit)
        amp_arr = gfit_amp_arr_from_args(args, n_lifetimes, n_wls)
        gfit_lifetimes = list(args[-n_lifetimes:])
        for i in range(n_wls):
            amps = list(amp_arr[:, i])
            exp_args = amps + gfit_lifetimes
            fitted[:, i] = multi_exp(ts_for_fit, *exp_args)
        return fitted.reshape(data_for_fit.shape[0] * data_for_fit.shape[1])
    
    res, _ = curve_fit(fit_me, xs, ys, p0=gfit_guesses, bounds=gfit_bounds)
    fit_amps = gfit_amp_arr_from_args(list(res), n_lifetimes, n_wls)
    fit_lifetimes = list(res[-n_lifetimes:])
    fits = np.empty_like(data)
    fits[ts <= fit_after, :] = 0
    for i in range(n_wls):
        exp_args = list(fit_amps[:, i]) + fit_lifetimes
        fits[ts > fit_after, i] = multi_exp(ts_for_fit, *exp_args)
    return fit_amps, fit_lifetimes