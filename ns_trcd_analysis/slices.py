import click
import h5py
import numpy as np
from typing import Union
from . import core


def valid_shot_slice_point(slice_time, slice_index, points) -> bool:
    """Ensure that an valid combination of slice time and/or slice index has been provided.
    """
    if (slice_time is None) and (slice_index is None):
        click.echo("Either a slice time or slice index is required. See the '--slice-time' or '--slice-index' options.", err=True)
        return False
    if (slice_time is not None) and (slice_index is not None):
        click.echo("Specify slice index or time, but not both", err=True)
        return False
    if (slice_index is not None) and ((slice_index < 0) or (slice_index >= (points - 1))):
        click.echo("Slice index out of range.")
        return False
    return True


def index_nearest_to_value(arr, value) -> Union[int, None]:
    """Return an index into a sorted array that's nearest to the provided value.

    This will return None if the value is outside the range of the array. The search
    is done via a linear scan.
    """
    if value < arr[0]:
        return None
    if value > arr[-1]:
        return None
    current_best_idx = -1
    current_best_diff = 1e9
    for i in range(len(arr)):
        diff = np.abs(value - arr[i])
        if diff < current_best_diff:
            current_best_diff = diff
            current_best_idx = i
        else:
            return current_best_idx


def abs_slice_at_index(infile, t_idx, wl_idx) -> np.ndarray:
    """Return an absorption slice along the shot axis.
    """
    ds = infile["data"]
    shots = ds.shape[1]
    slice_values = np.empty(shots)
    par = ds[t_idx, 0, :, wl_idx, 0]
    ref = ds[t_idx, 2, :, wl_idx, 0]
    absorption = -np.log10(par / ref)
    return absorption