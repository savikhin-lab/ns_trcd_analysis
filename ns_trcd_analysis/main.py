import click
import h5py
import numpy as np
from pathlib import Path
from . import core
from . import delta_a
from . import raw2hdf5


POINTS = 20_000


@click.group()
def cli():
    pass


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("outfile_name", type=click.Path(file_okay=True, dir_okay=False))
def assemble(input_dir, outfile_name):
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


cli.add_command(assemble)
cli.add_command(da)
