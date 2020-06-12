import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Channels(Enum):
    PAR = 0
    PERP = 1
    REF = 2


CHANNEL_MAP = {
    "par": Channels.PAR,
    "perp": Channels.PERP,
    "ref": Channels.REF,
}


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
            count += 1
    return count


def save_txt(arr, path) -> None:
    """Save a CSV file of dA or dCD data.
    """
    np.savetxt(path, arr, delimiter=",")


def save_fig(x, y, path, xlabel=None, ylabel=None) -> None:
    """Save a PNG image of dA or dCD data.

    Neither the x nor y data will be modified for plotting, so you will need to convert
    to microseconds or mOD before passing data to this function.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=200)
    plt.close()


def time_axis(tpp=20e-9, length=20_000) -> np.ndarray:
    """Return the time axis used in experiments.
    """
    ts = tpp * np.arange(length)
    ts -= 0.1 * ts[-1]
    return ts
