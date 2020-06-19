import h5py
import numpy as np
from typing import Union
from . import core


def valid_shot_slice_point(slice_time, slice_index) -> bool:
    """Ensure that an valid combination of slice time and/or slice index has been provided.
    """
    if (slice_time is None) and (slice_index is None):
        click.echo("Either a slice time or slice index is required. See the '--slice-time' or '--slice-index' options.", err=True)
        return False
    if (slice_time is not None) and (slice_index is not None):
        click.echo("Specify slice index or time, but not both", err=True)
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


def raw_slice_at_time(infile, channel, t) -> Union[np.ndarray, None]:
    """Return a slice along the shot-axis for the given time and channel.
    """
    ts = core.time_axis()
    slice_index = index_nearest_to_value(ts, t)
    if slice_index is None:
        return None
    dataset = infile["data"]
    _, _, num_shots, _, _ = dataset.shape
    slice_values = np.empty(num_shots)
    for shot in range(num_shots):
        slice_values[shot] = dataset[slice_index, channel.value, shot, 0, 0]
    return slice_values


def raw_slice_at_index(infile, channel, idx) -> Union[np.ndarray, None]:
    """Return a slice along the shot-axis for the given channel and index along the time axis.
    """
    dataset = infile["data"]
    points, _, num_shots, _, _ = dataset.shape
    if idx < 0:
        return None
    if idx > points:
        return None
    slice_values = np.empty(num_shots)
    for shot in range(num_shots):
        slice_values[shot] = dataset[idx, channel.value, shot, 0, 0]
    return slice_values


def da_slice_at_time(infile, t) -> Union[np.ndarray, None]:
    """Return a slice along the shot-axis at the given time.
    """
    ts = core.time_axis()
    slice_index = index_nearest_to_value(ts, t)
    if slice_index is None:
        return None
    dataset = infile["data"]
    _, num_shots, _ = dataset.shape
    slice_values = np.empty(num_shots)
    for shot in range(num_shots):
        slice_values[shot] = dataset[slice_index, shot, 0]
    return slice_values


def da_slice_at_index(infile, idx) -> Union[np.ndarray, None]:
    """Return a slice along the shot-axis at the given index along the time axis.
    """
    dataset = infile["data"]
    points, num_shots, _ = dataset.shape
    if idx < 0:
        return None
    if idx > points:
        return None
    slice_values = np.empty(num_shots)
    for shot in range(num_shots):
        slice_values[shot] = dataset[idx, shot, 0]
    return slice_values
