# The shape of the input dataset is as follows:
# (points, channels, shots, wavelengths, pump states)
#
# The shape of the output dataset is as follows:
# (points, shots, wavelengths)
#
import click
import numpy as np
from scipy.optimize import curve_fit


BEFORE_ZERO_POINTS = 1_500


def compute_da(input_ds, output_ds, incremental):
    """Compute dA from the raw parallel and reference channels.
    """
    _, _, shots, wavelengths, _ = input_ds.shape
    if not incremental:
        tmp_raw = np.empty((20_000, 3, shots, wavelengths, 1))
        tmp_da = np.empty((20_000, shots, wavelengths))
        input_ds.read_direct(tmp_raw)
    with click.progressbar(range(shots), label="Computing dA") as shots:
        for shot_idx in shots:
            for wl_idx in range(wavelengths):
                if incremental:
                    par = input_ds[:, 0, shot_idx, wl_idx, 0]
                    ref = input_ds[:, 2, shot_idx, wl_idx, 0]
                    before_zero_par = par[:BEFORE_ZERO_POINTS]
                    before_zero_ref = ref[:BEFORE_ZERO_POINTS]
                    without_pump = np.mean(before_zero_par / before_zero_ref)
                    output_ds[:, shot_idx, wl_idx] = -np.log10(par / ref / without_pump)
                else:
                    par = tmp_raw[:, 0, shot_idx, wl_idx, 0]
                    ref = tmp_raw[:, 2, shot_idx, wl_idx, 0]
                    before_zero_par = par[:BEFORE_ZERO_POINTS]
                    before_zero_ref = ref[:BEFORE_ZERO_POINTS]
                    without_pump = np.mean(before_zero_par / before_zero_ref)
                    tmp_da[:, shot_idx, wl_idx] = -np.log10(par / ref / without_pump)
    if not incremental:
        output_ds.write_direct(tmp_da)
    return


def average(f, incremental) -> np.ndarray:
    """Average all measurements for each wavelength.
    """
    da_ds = f["data"]
    points, shots, wls = da_ds.shape
    avg_ds = f.create_dataset("average", (points, wls))
    avg_ds.write_direct(np.mean(da_ds, axis=1))
    return


def subtract_background(dataset) -> None:
    """Subtract a linear background from a set of dA curves.
    """
    points, shots, wls = dataset.shape
    x = np.arange(points)
    x_before_zero = x[:BEFORE_ZERO_POINTS]
    for shot_num in range(shots):
        for wl_num in range(wls):
            da_before_zero = dataset[:BEFORE_ZERO_POINTS, shot_num, wl_num]
            (slope, intercept), _ = curve_fit(line, x_before_zero, da_before_zero)
            background = line(x, slope, intercept)
            dataset[:, shot_num, wl_num] -= background
    return


def line(x, m, b) -> np.ndarray:
    """Compute a line for use with background subtraction.
    """
    return m * x + b
