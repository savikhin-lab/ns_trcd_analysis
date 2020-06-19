import click
import numpy as np
import matplotlib.pyplot as plt
from . import core


def dump_raw_images(path, channel, arr, wl_idx) -> None:
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
            outfile = path / f"{shot_num+1:03d}.png"
            core.save_fig(ts, arr[:, channel.value, shot_num, wl_idx, 0], outfile)
    return


def dump_da_images(path, arr, wl_idx) -> None:
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
            outfile = path / f"{shot_num+1:03d}.png"
            core.save_fig(ts, arr[:, shot_num, wl_idx], outfile)
    return
