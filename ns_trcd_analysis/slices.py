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
    if (slice_index is not None) and ((slice_index < 0) or (slice_index >= 20_000)):
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


def raw_slice_at_index(infile, channel, t_idx, wl_idx) -> np.ndarray:
    """Return a slice along the shot-axis for the given channel and index along the time axis.
    """
    dataset = infile["data"]
    points, _, num_shots, _, _ = dataset.shape
    slice_values = np.empty(num_shots)
    for shot_idx in range(num_shots):
        slice_values[shot_idx] = dataset[t_idx, channel.value, shot_idx, wl_idx, 0]
    return slice_values


def da_slice_at_index(infile, t_idx, wl_idx) -> np.ndarray:
    """Return a slice along the shot-axis at the given index along the time axis.
    """
    dataset = infile["data"]
    points, num_shots, _ = dataset.shape
    slice_values = np.empty(num_shots)
    for shot_idx in range(num_shots):
        slice_values[shot_idx] = dataset[t_idx, shot_idx, wl_idx]
    return slice_values