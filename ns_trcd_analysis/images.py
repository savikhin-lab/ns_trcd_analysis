import click
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from . import core


class Channels(Enum):
    PAR = 0
    PERP = 1
    REF = 2


def dump_raw_images(path, channel, arr) -> None:
    """Generate plots of each shot for a particular channel.
    """
    try:
        _, _, num_shots, _, _ = arr.shape
    except ValueError:
        click.echo(f"Incorrect data format: {len(arr.shape)} dimensions when 5 are expected.", err=True)
        return
    ts = core.time_axis()
    if not path.exists():
        path.mkdir()
    with click.progressbar(range(num_shots), label="Generating images") as shots:
        for shot_num in shots:
            outfile = path / (str(shot_num) + ".png")
            core.save_fig(ts, arr[:, channel.value, shot_num, 0, 0], outfile)
    return


def dump_da_images(path, arr) -> None:
    """Generate plots for each dA measurement.
    """
    try:
        _, num_shots, _ = arr.shape
    except ValueError:
        click.echo(f"Incorrect data format: {len(arr.shape)} dimensions when 3 are expected.", err=True)
        return
    ts = core.time_axis()
    if not path.exists():
        path.mkdir()
    with click.progressbar(range(num_shots), label="Generating images") as shots:
        for shot_num in shots:
            outfile = path / (str(shot_num) + ".png")
            core.save_fig(ts, arr[:, shot_num, 0], outfile)
    return
