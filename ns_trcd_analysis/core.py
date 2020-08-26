import itertools
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Union


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
        chan = CHANNEL_MAP[channel_str]
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
        devs = np.abs((y - mean)/std_dev)
        for i in range(len(y)):
            if devs[i] > 2:
                y[i] = (y[i-2] + y[i+2])/2
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y)
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
    ten_percent_point = np.floor(length/10) * tpp
    ts -= ten_percent_point
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
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if len(chunk) == 0:
            break
        yield chunk