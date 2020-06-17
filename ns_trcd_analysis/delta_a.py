# The shape of the input dataset is as follows:
# (points, channels, shots, wavelengths, pump states)
#
# The shape of the output dataset is as follows:
# (points, shots, wavelengths)
#
import numpy as np
from scipy.optimize import curve_fit


BEFORE_ZERO_POINTS = 1_500


def compute_da(input_ds, output_ds):
    """Compute dA from the raw parallel and reference channels.
    """
    _, _, shots, wavelengths, _ = input_ds.shape
    for shot_num in range(shots):
        for wl_num in range(wavelengths):
            par = input_ds[:, 0, shot_num, wl_num, 0]
            ref = input_ds[:, 2, shot_num, wl_num, 0]
            before_zero_par = par[:BEFORE_ZERO_POINTS]
            before_zero_ref = ref[:BEFORE_ZERO_POINTS]
            without_pump = np.mean(before_zero_par / before_zero_ref)
            da = -np.log10(par / ref / without_pump)
            output_ds[:, shot_num, wl_num] = da
    return


def average(ds) -> np.ndarray:
    """Average all measurements for each wavelength.
    """
    return np.mean(ds, axis=1).reshape(20_000)


def subtract_background(dataset) -> None:
    """Subtract a linear background from a set of dA curves.
    """
    points, shots, _ = dataset.shape
    x = np.arange(points)
    x_before_zero = x[:BEFORE_ZERO_POINTS]
    for shot_num in range(shots):
        da_before_zero = dataset[:BEFORE_ZERO_POINTS, shot_num, 0]
        (slope, intercept), _ = curve_fit(line, x_before_zero, da_before_zero)
        background = line(x, slope, intercept)
        dataset[:, shot_num, 0] -= background
    return


def line(x, m, b) -> np.ndarray:
    """Compute a line for use with background subtraction.
    """
    return m * x + b
