import click
import h5py
import numpy as np
from pathlib import Path
from . import core
from . import delta_a
from . import images
from . import raw2hdf5
from . import slices
from .core import Channels, valid_channel


POINTS = 20_000


@click.group()
def cli():
    pass


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("outfile_name", type=click.Path(file_okay=True, dir_okay=False))
@click.option("-i", "--incremental", is_flag=True, help="Write shots to the file channel by channel, rather than shot by shot.")
def assemble(input_dir, outfile_name, incremental):
    """Read a directory of experiment data into an HDF5 file.

    \b
    The format of the input directory should be as follows:
    <input dir>
        <shot dir> (one for each shot)
            par.npy
            perp.npy
            ref.npy

    The resulting HDF5 file will have a dataset called 'data' which has the following shape:
    (<points>, <channels>, <shots>, <wavelengths>, <pump states>)

    For the moment <pump states> is 1 and thus doesn't need to be there, but is included for backwards compatibility.
    """
    in_dir = Path(input_dir)
    outfile = in_dir / outfile_name
    raw2hdf5.ingest(in_dir, outfile, incremental)


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_file", type=click.Path(file_okay=True, dir_okay=False))
@click.option("-a", "--average", is_flag=True, help="Average dA and save the result.")
@click.option("-s", "--subtract-background", is_flag=True, help="Subtract a linear background from dA.")
@click.option("-f", "--figure-path", "fig", type=click.Path(file_okay=True, dir_okay=False), help="Save a figure of the average dA. Only valid with the '-a' option.")
@click.option("-t", "--save-txt-path", "txt", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV of the average dA. Only valid with the '-a' option.")
def da(input_file, output_file, average, subtract_background, fig, txt):
    """Compute dA from a raw data file.

    The output is stored in a separate file (OUTPUT_FILE) with the shape (points, shots, wavelengths).
    """
    with h5py.File(output_file, "w") as outfile:
        with h5py.File(input_file, "r") as infile:
            (points, channels, shots, wavelengths, pump_states) = infile["data"].shape
            outfile.create_dataset("data", (points, shots, wavelengths))
            delta_a.compute_da(infile["data"], outfile["data"])
            if subtract_background:
                delta_a.subtract_background(outfile["data"])
            if average:
                avg = delta_a.average(outfile["data"])
                ts = core.time_axis()
                if txt:
                    outdata = np.empty((POINTS, 2))
                    outdata[:, 0] = ts
                    outdata[:, 1] = avg
                    core.save_txt(outdata, txt)
                if fig:
                    core.save_fig(ts, avg, fig)
            else:
                if txt:
                    click.echo("Saving a CSV requires averaging ('--average').", err=True)
                    return
                if fig:
                    click.echo("Saving an image requires averaging ('--average').", err=True)
                    return
    return


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("-d", "--data-format", "format", type=click.Choice(["raw", "da"]), help="The format of the data file.")
@click.option("-c", "--channel", type=click.Choice(["par", "perp", "ref"]), help="If the format of the data is 'raw', which channel to inspect.")
@click.option("-w", "--wavelength", type=click.INT, help="The wavelength to inspect.")
def inspect(input_file, output_dir, format, channel, wavelength):
    """Generate images of each shot in a data file.

    This works for both dA and raw data files (specified with the '-d' flag).
    """
    if wavelength is None:
        click.echo("A wavelength is required. See the '-w' option.")
        return
    with h5py.File(input_file, "r") as infile:
        wl_idx = core.index_for_wavelength(list(infile["wavelengths"]), wavelength)
        if wl_idx is None:
            click.echo("Wavelength not found.")
            return
        dataset = infile["data"]
        root_dir = Path(output_dir)
        if format == "da":
            images.dump_da_images(root_dir, dataset, wl_idx)
        elif format == "raw":
            if not channel:
                click.echo("Raw data format requires a channel specifier. See the '-c' option.", err=True)
                return
            if channel == "par":
                images.dump_raw_images(root_dir, Channels.PAR, dataset, wl_idx)
            elif channel == "perp":
                images.dump_raw_images(root_dir, Channels.PERP, dataset, wl_idx)
            elif channel == "ref":
                images.dump_raw_images(root_dir, Channels.REF, dataset, wl_idx)
            else:
                click.echo("Invalid channel or incorrect data format", err=True)
                return
        else:
            click.echo("Invalid data format", err=True)
            return


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("-d", "--data-format", "format", type=click.Choice(["raw", "da"]), help="The format of the data file.")
@click.option("-c", "--channel", type=click.Choice(["par", "perp", "ref"]), help="If the format of the data is 'raw', which channel to slice.")
@click.option("-f", "--figure-path", "figpath", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txtpath", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("--slice-time", "stime", type=click.FLOAT, help="Select the slice closest to the specified time.")
@click.option("--slice-index", "sindex", type=click.INT, help="Select the slice at the specified index along the time axis.")
def shotslice(input_file, format, channel, figpath, txtpath, stime, sindex):
    """Select the same point in time for every shot in the dataset at a fixed wavelength.
    """
    if format is None:
        click.echo("A format specifier is required. See the '-d' option.", err=True)
        return
    if not slices.valid_shot_slice_point(stime, sindex):
        return
    if (txtpath is None) and (figpath is None):
        click.echo("No output has been chosen. See '-f' or '-t'.", err=True)
        return
    if format == "raw":
        if not core.valid_channel(channel):
            return
        chan = core.CHANNEL_MAP[channel]
        if stime is not None:
            s = slices.raw_slice_at_time(input_file, chan, stime)
        else:
            s = slices.raw_slice_at_index(input_file, chan, sindex)
    elif format == "da":
        if channel is not None:
            click.echo("Channel specifiers are only valid for the 'raw' data format.", err=True)
            return
        if stime is not None:
            s = slices.da_slice_at_time(input_file, stime)
        else:
            s = slices.da_slice_at_index(input_file, sindex)
    if s is None:
        click.echo("Slice falls outside the range of experimental data.", err=True)
        return
    shots = np.arange(len(s))
    if txtpath:
        txtdata = np.empty((len(shots), 2))
        txtdata[:, 0] = shots
        txtdata[:, 1] = s
        core.save_txt(txtdata, txtpath)
    if figpath:
        core.save_fig(shots, s, figpath)
    return


cli.add_command(assemble)
cli.add_command(da)
cli.add_command(inspect)
cli.add_command(shotslice)
