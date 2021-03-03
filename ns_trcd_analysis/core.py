import itertools
import click
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path
from typing import Union, List, Tuple


class Channels(Enum):
    PAR = 0
    PERP = 1
    REF = 2


CHANNEL_MAP = {
    "par": Channels.PAR,
    "perp": Channels.PERP,
    "ref": Channels.REF,
}

POINTS = 20_000
VALENTYN_POINTS = 50_000


def valid_channel(channel_str) -> bool:
    """Determine whether a string represents a valid channel.
    """
    if channel_str is None:
        click.echo("A channel specifier is required when the data format is 'raw'.", err=True)
        return False
    try:
        _ = CHANNEL_MAP[channel_str]
    except KeyError:
        click.echo("Invalid channel name.", err=True)
        return False
    return True


def count_subdirs(path) -> int:
    """Count the number of subdirectories directly under `path`.

    This is useful for determining the number of wavelengths and shots in an experiment.
    """
    count = 0
    for item in path.iterdir():
        if item.is_dir():
            if item.name[0] == "_":
                continue
            count += 1
    return count


def save_txt(arr, path) -> None:
    """Save a CSV file of dA or dCD data.
    """
    np.savetxt(path, arr, delimiter=",")


def save_fig(x, y, path, xlabel=None, ylabel=None, title=None, remove_dev=False) -> None:
    """Save a PNG image of dA or dCD data.

    Neither the x nor y data will be modified for plotting, so you will need to convert
    to microseconds or mOD before passing data to this function.
    """
    if remove_dev:
        mean = np.mean(y)
        std_dev = np.std(y)
        devs = np.abs((y - mean) / std_dev)
        for i in range(len(y)):
            if devs[i] > 2:
                y[i] = (y[i - 2] + y[i + 2]) / 2
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y, linewidth=0.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=200)
    plt.close()


def time_axis(tpp=20e-9, length=20_000) -> np.ndarray:
    """Return the time axis used in experiments.
    """
    ts = tpp * np.arange(length)
    ten_percent_point = np.floor(length / 10) * tpp
    ts -= ten_percent_point
    ts *= 1e6  # convert from seconds to microseconds
    return ts


def index_for_wavelength(wls, w) -> Union[None, int]:
    """Return the index for a particular wavelength or None if not present
    """
    try:
        idx = wls.index(w)
    except ValueError:
        return None
    return idx


def iter_chunks(iterable, size):
    """Returns chunks of an iterable at a time.
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if len(chunk) == 0:
            break
        yield chunk


def compute_splits(points, size) -> List[Tuple[int, int]]:
    """Compute index ranges of a given size from a maximum number of indices.
    """
    splits = []
    points_left = points
    cursor = 0
    while True:
        if points_left > size:
            splits.append((cursor, cursor + size))
            cursor += size
            points_left -= size
            continue
        if points_left == 0:
            break
        else:
            splits.append((cursor, cursor + points_left))
        break
    return splits


def load_dir_into_arr(d: Path) -> (np.ndarray, np.ndarray):
    """Load the text files in the given directory into an array.
    """
    files = sorted([f for f in d.iterdir() if f.suffix == ".txt"])
    first = np.loadtxt(files[0], delimiter=",")
    arr = np.empty((first.shape[0], len(files)))
    for i, f in enumerate(files):
        arr[:, i] = np.loadtxt(f, delimiter=",")[:, 1]
    return arr, first[:, 0]
