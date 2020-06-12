import click
import h5py
import numpy as np
from pathlib import Path
from . import core
from . import delta_a
from . import images
from . import raw2hdf5
from .images import Channels


POINTS = 20_000


@click.group()
def cli():
    pass


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("outfile_name", type=click.Path(file_okay=True, dir_okay=False))
def assemble(input_dir, outfile_name):
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

    For the moment both <wavelengths> and <pump states> are 1 and thus don't need to be there, but are included for backwards compatibility.
    """
    in_dir = Path(input_dir)
    outfile = in_dir / outfile_name
    raw2hdf5.ingest(in_dir, outfile)


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
def inspect(input_file, output_dir, format, channel):
    """Generate images of each shot in a data file.

    This works for both dA and raw data files (specified with the '-d' flag).
    """
    with h5py.File(input_file, "r") as infile:
        dataset = infile["data"]
        root_dir = Path(output_dir)
        if format == "da":
            images.dump_da_images(root_dir, dataset)
        elif format == "raw":
            if not channel:
                click.echo("Raw data format requires a channel specifier. See the '-c' option.", err=True)
                return
            if channel == "par":
                images.dump_raw_images(root_dir, Channels.PAR, dataset)
            elif channel == "perp":
                images.dump_raw_images(root_dir, Channels.PERP, dataset)
            elif channel == "ref":
                images.dump_raw_images(root_dir, Channels.REF, dataset)
            else:
                click.echo("Invalid channel or incorrect data format", err=True)
                return
        else:
            click.echo("Invalid data format", err=True)
            return


cli.add_command(assemble)
cli.add_command(da)
cli.add_command(inspect)
