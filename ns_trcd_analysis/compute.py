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


def compute_da(infile, outfile):
    """Compute dA from the raw parallel and reference channels.
    """
    raw_ds = infile["data"]
    _, _, shots, wavelengths, _ = raw_ds.shape
    tmp_raw = np.empty((20_000, 3, shots, wavelengths, 1))
    tmp_da = np.empty((20_000, shots, wavelengths))
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


def compute_perp_da(infile, outfile):
    """Compute dA from the raw perpendicular and reference channels.
    """
    raw_ds = infile["data"]
    _, _, shots, wavelengths, _ = raw_ds.shape
    tmp_raw = np.empty((20_000, 3, shots, wavelengths, 1))
    tmp_da = np.empty((20_000, shots, wavelengths))
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


def compute_cd_approx(infile, outfile, delta):
    """Compute dCD from the rawl parallel and perpendicular channels.
    """
    ds_in = infile["data"]
    _, _, shots, wavelengths, _ = ds_in.shape
    tmp_raw = np.empty((20_000, 3, shots, wavelengths, 1))
    tmp_cd = np.empty((20_000, shots, wavelengths))
    ds_in.read_direct(tmp_raw)
    coeff = 4 / (2.3 * delta)
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


def compute_cd_exact(infile, outfile, delta):
    """Compute the exact dCD via solving a system of equations.
    """
    pass


def average(f) -> np.ndarray:
    """Average all measurements for each wavelength.
    """
    da_ds = f["data"]
    points, shots, wls = da_ds.shape
    avg_ds = f.create_dataset("average", (points, wls))
    avg_ds.write_direct(np.mean(da_ds, axis=1))
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


def save_avg_as_txt(f, outdir):
    """Save the average dA for each wavelength as a CSV file.
    """
    ts = core.time_axis()
    da = f["average"]
    _, wls = da.shape
    outdata = np.empty((20_000, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving CSVs") as indices:
        for wl_idx in indices:
            outdata[:, 1] = da[:, wl_idx]
            outpath = outdir / f"{wavelengths[wl_idx]}.txt"
            core.save_txt(outdata, outpath)
    return


def save_avg_as_png(f, outdir, xlabel=None, ylabel=None, title=None):
    """Save the average dA for each wavelength as a PNG file.
    """
    ts = core.time_axis()
    da = f["average"]
    _, wls = da.shape
    outdata = np.empty((20_000, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving figures") as indices:
        for wl_idx in indices:
            outpath = outdir / f"{wavelengths[wl_idx]}.png"
            core.save_fig(ts, da[:, wl_idx], outpath, remove_dev=True)
    return


def save_da_figures(f, outdir):
    """Save the average dA for each wavelength as a PNG file.
    """
    ts = core.time_axis()
    da = f["average"]
    _, wls = da.shape
    outdata = np.empty((20_000, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving figures") as indices:
        for wl_idx in indices:
            outpath = outdir / f"{wavelengths[wl_idx]}.png"
            core.save_fig(ts*1_000_000, da[:, wl_idx]*1_000, outpath, xlabel="Time (us)", ylabel="dA (mOD)", title=f"{wavelengths[wl_idx]}nm", remove_dev=True)
    return


def save_cd_figures(f, outdir):
    """Save the average dA for each wavelength as a PNG file.
    """
    ts = core.time_axis()
    cd = f["average"]
    _, wls = cd.shape
    outdata = np.empty((20_000, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving figures") as indices:
        for wl_idx in indices:
            outpath = outdir / f"{wavelengths[wl_idx]}.png"
            core.save_fig(ts*1_000_000, cd[:, wl_idx]*1_000, outpath, xlabel="Time (us)", ylabel="dCD", title=f"{wavelengths[wl_idx]}nm", remove_dev=True)
    return


def local_fits(infile, lifetimes) -> Dict[int, List[Tuple[float, float]]]:
    """Compute local fits for a set of dA or dCD curves.
    """
    raw_data = np.empty_like(infile["average"])
    infile["average"].read_direct(raw_data)
    params = []
    for x in lifetimes:
        params.append(-1e-3)
        params.append(x)
    wavelengths = infile["wavelengths"]
    ts = core.time_axis()[FIT_START_POINT:]
    lfit_results = dict()
    with click.progressbar(range(len(wavelengths)), label="Computing fits") as indices:
        for wl_idx in indices:
            y_data = raw_data[FIT_START_POINT:, wl_idx]
            optimized_params, cov = curve_fit(multi_exp, ts, raw_data[FIT_START_POINT:, wl_idx], p0=params, maxfev=10_000)
            pairs = [x for x in core.iter_chunks(optimized_params, 2)]
            lfit_results[wavelengths[wl_idx]] = pairs
    return lfit_results


def multi_exp(x, *args) -> np.ndarray:
    """Compute a multi-exponential decay function.

    The first argument is the time axis. The arguments that follow must be in the
    order 'a1', 't1', 'a2', 't2', ... , 'an', 'tn'.
    """
    out = np.zeros_like(x)
    for a, tau in core.iter_chunks(args, 2):
        this_exp = a * np.exp(-x/tau)
        np.add(out, this_exp, out=out)
    return out


def save_lfit_params_as_txt(results, outfile):
    """Save the local fit amplitudes and lifetimes to a text file.
    """
    for wl in results.keys():
        outfile.write(f"[{wl:3d}]\n")
        count = 1
        for a, t in results[wl]:
            t_us = t * 1_000_000
            outfile.write(f"A{count}: {a:.2e}\n")
            outfile.write(f"T{count}: {t_us:.2f}us\n")
            count += 1
        outfile.write("\n")
    return