import click
import h5py
import numpy as np
from pathlib import Path
from . import core
from . import compute
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

    For the moment <pump states> is 1 and thus doesn't need to be there, but is included for backwards compatibility.
    """
    in_dir = Path(input_dir)
    outfile = in_dir / outfile_name
    raw2hdf5.ingest(in_dir, outfile)


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_file", type=click.Path(file_okay=True, dir_okay=False))
@click.option("-a", "--average", is_flag=True, help="Average dA and save the result.")
@click.option("-s", "--subtract-background", is_flag=True, help="Subtract a linear background from dA.")
@click.option("-f", "--figure-path", "fig", type=click.Path(file_okay=False, dir_okay=True), help="Save a figure of the average dA. Only valid with the '-a' option.")
@click.option("-t", "--save-txt-path", "txt", type=click.Path(file_okay=False, dir_okay=True), help="Save a CSV of the average dA. Only valid with the '-a' option.")
@click.option("-p", "--perp", is_flag=True, help="Compute dA with the perpendicular channel rather than parallel.")
def da(input_file, output_file, average, subtract_background, fig, txt, perp):
    """Compute dA from a raw data file.

    The output is stored in a separate file (OUTPUT_FILE) with the shape (points, shots, wavelengths).
    """
    with h5py.File(output_file, "w") as outfile:
        with h5py.File(input_file, "r") as infile:
            (points, channels, shots, wavelengths, pump_states) = infile["data"].shape
            outfile.create_dataset("data", (points, shots, wavelengths))
            outfile.create_dataset("wavelengths", (wavelengths,), data=infile["wavelengths"])
            if perp:
                compute.compute_perp_da(infile, outfile)
            else:
                compute.compute_da(infile, outfile)
            if subtract_background:
                compute.subtract_background(outfile)
            if average:
                compute.average(outfile)
                if txt:
                    compute.save_avg_as_txt(outfile, Path(txt))
                if fig:
                    compute.save_da_figures(outfile, Path(fig))
            else:
                if txt:
                    click.echo("Saving a CSV requires averaging. See the '-a' option.", err=True)
                    return
                if fig:
                    click.echo("Saving an image requires averaging. See the '-a' option.", err=True)
                    return
    return


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_file", type=click.Path(file_okay=True, dir_okay=False))
@click.option("-d", "--delta", type=click.FLOAT, required=True, help="The value of delta to use when computing dCD.")
@click.option("-a", "--average", is_flag=True, help="Average dA and save the result.")
@click.option("-s", "--subtract-background", is_flag=True, help="Subtract a linear background from dA.")
@click.option("-f", "--figure-path", "fig", type=click.Path(file_okay=False, dir_okay=True), help="Save a figure of the average dA. Only valid with the '-a' option.")
@click.option("-t", "--save-txt-path", "txt", type=click.Path(file_okay=False, dir_okay=True), help="Save a CSV of the average dA. Only valid with the '-a' option.")
def cd(input_file, output_file, delta, average, subtract_background, fig, txt):
    """Compute dCD from a raw data file.

    The output is stored in a separate file (OUTPUT_FILE) with the shape (points, shots, wavelengths).
    """
    with h5py.File(output_file, "w") as outfile:
        with h5py.File(input_file, "r") as infile:
            (points, channels, shots, wavelengths, pump_states) = infile["data"].shape
            outfile.create_dataset("data", (points, shots, wavelengths))
            outfile.create_dataset("wavelengths", (wavelengths,), data=infile["wavelengths"])
            compute.compute_cd_approx(infile, outfile, delta)
            if subtract_background:
                compute.subtract_background(outfile)
            if average:
                compute.average(outfile)
                if txt:
                    compute.save_avg_as_txt(outfile, Path(txt))
                if fig:
                    compute.save_cd_figures(outfile, Path(fig))
            else:
                if txt:
                    click.echo("Saving a CSV requires averaging. See the '-a' option.", err=True)
                    return
                if fig:
                    click.echo("Saving an image requires averaging. See the '-a' option.", err=True)
                    return
    return


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("-d", "--data-format", "format", type=click.Choice(["raw", "da"]), help="The format of the data file.")
@click.option("-c", "--channel", type=click.Choice(["par", "perp", "ref"]), help="If the format of the data is 'raw', which channel to inspect.")
@click.option("-w", "--wavelength", type=click.INT, required=True, help="The wavelength to inspect.")
def inspect(input_file, output_dir, format, channel, wavelength):
    """Generate images of each shot in a data file.

    This works for both dA and raw data files (specified with the '-d' flag).
    """
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
            chan = core.CHANNEL_MAP[channel]
            images.dump_raw_images(root_dir, chan, dataset, wl_idx)
        else:
            click.echo("Invalid data format", err=True)
            return


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("-d", "--data-format", "format", type=click.Choice(["raw", "da"]), required=True, help="The format of the data file.")
@click.option("-c", "--channel", type=click.Choice(["par", "perp", "ref"]), required=True, help="If the format of the data is 'raw', which channel to slice.")
@click.option("-f", "--figure-path", "figpath", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txtpath", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("--slice-time", "stime", type=click.FLOAT, help="Select the slice closest to the specified time.")
@click.option("--slice-index", "sindex", type=click.INT, help="Select the slice at the specified index along the time axis.")
@click.option("-w", "--wavelength", type=click.INT, required=True, help="The wavelength to create a slice of.")
def shotslice(input_file, format, channel, figpath, txtpath, stime, sindex, wavelength):
    """Select the same point in time for every shot in the dataset at a fixed wavelength.
    """
    with h5py.File(input_file, "r") as infile:
        if (txtpath is None) and (figpath is None):
            click.echo("No output has been chosen. See '-f' or '-t'.", err=True)
            return
        points = infile["data"].shape[0]
        if not slices.valid_shot_slice_point(stime, sindex, points):
            return
        if sindex is None:
            s_idx = slices.index_nearest_to_value(core.time_axis(), stime)
            if s_idx is None:
                click.echo("Slice time is out of range.")
                return
        else:
            s_idx = sindex
        wl_idx = core.index_for_wavelength(list(infile["wavelengths"]), wavelength)
        if wl_idx is None:
            click.echo("Wavelength not found.")
            return
        if format == "raw":
            if not core.valid_channel(channel):
                return
            chan = core.CHANNEL_MAP[channel]
            s = infile["data"][s_idx, chan.value, :, wl_idx, 0]
        elif format == "da":
            if channel is not None:
                click.echo("Channel specifiers are only valid for the 'raw' data format.", err=True)
                return
            s = infile["data"][s_idx, :, wl_idx]
        shots = np.arange(len(s))
        if txtpath:
            txtdata = np.empty((len(shots), 2))
            txtdata[:, 0] = shots
            txtdata[:, 1] = s
            core.save_txt(txtdata, txtpath)
        if figpath:
            t = core.time_axis()[s_idx] * 1_000_000
            core.save_fig(shots, s, figpath, xlabel="Shot Number", title=f"{wavelengt}nm, t={t:.2f}us")
    return


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("-f", "--figure-path", "figpath", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txtpath", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("--slice-time", "stime", type=click.FLOAT, help="Select the slice closest to the specified time.")
@click.option("--slice-index", "sindex", type=click.INT, help="Select the slice at the specified index along the time axis.")
def wlslice(input_file, figpath, txtpath, stime, sindex):
    """Create a dA slice at all wavelengths for a specified time.

    Note: This command is only valid for averaged dA data.
    """
    with h5py.File(input_file, "r") as infile:
        try:
            infile["average"]
        except KeyError:
            click.echo("This command is only valid for averaged data.")
            return
        if (txtpath is None) and (figpath is None):
            click.echo("No output has been chosen. See '-f' or '-t'.", err=True)
            return
        points = infile["data"].shape[0]
        if not slices.valid_shot_slice_point(stime, sindex, points):
            return
        if sindex is None:
            s_idx = slices.index_nearest_to_value(core.time_axis(), stime)
            if s_idx is None:
                click.echo("Slice time is out of range.")
                return
        else:
            s_idx = sindex
        s = infile["average"][s_idx, :]
        wavelengths = infile["wavelengths"]
        if txtpath:
            txtdata = np.empty((len(wavelengths), 2))
            txtdata[:, 0] = wavelengths
            txtdata[:, 1] = s
            core.save_txt(txtdata, txtpath)
        if figpath:
            t = core.time_axis()[s_idx] * 1_000_000
            core.save_fig(wavelengths, s * 1_000, figpath, xlabel="Wavelength", ylabel="dA (mOD)", title=f"Slice at t={t:.2f}us")
        return


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("-f", "--figure-path", "figpath", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txtpath", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("--slice-time", "stime", type=click.FLOAT, help="Select the slice closest to the specified time.")
@click.option("--slice-index", "sindex", type=click.INT, help="Select the slice at the specified index along the time axis.")
@click.option("-w", "--wavelength", type=click.INT, required=True, help="The wavelength to create a slice of.")
def absslice(input_file, figpath, txtpath, stime, sindex, wavelength):
    """Create a slice of the absorption for a specific time and wavelength.

    Note: This command is only valid for raw data.
    """
    with h5py.File(input_file, "r") as infile:
        if len(infile["data"].shape) != 5:
            click.echo("This command only works with raw data. (Incorrect number of dimensions).")
            return
        if (txtpath is None) and (figpath is None):
            click.echo("No output has been chosen. See '-f' or '-t'.", err=True)
            return
        points = infile["data"].shape[0]
        if not slices.valid_shot_slice_point(stime, sindex, points):
            return
        if sindex is None:
            s_idx = slices.index_nearest_to_value(core.time_axis(), stime)
            if s_idx is None:
                click.echo("Slice time is out of range.")
                return
        else:
            s_idx = sindex
        wl_idx = core.index_for_wavelength(list(infile["wavelengths"]), wavelength)
        if wl_idx is None:
            click.echo("Wavelength not found.")
            return
        s = slices.abs_slice_at_index(infile, s_idx, wl_idx)
        shots = np.arange(len(s))
        if txtpath:
            txtdata = np.empty((len(shots), 2))
            txtdata[:, 0] = shots
            txtdata[:, 1] = s
            core.save_txt(txtdata, txtpath)
        if figpath:
            t = core.time_axis()[s_idx] * 1_000_000
            core.save_fig(shots, s, figpath, xlabel="Shot number", ylabel="Abs.", title=f"Slice at {wavelength}nm, t={t:.2f}us")
        return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The input file containing either dA or dCD data.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The output file in which to store the lifetimes and amplitudes.")
@click.option("-f", "--figure-path", "figpath", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txtpath", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("-l", "--lifetime", "lifetimes", type=click.FLOAT, multiple=True, required=True, help="The initial guesses for each lifetime. Multiple instances of this option are allowed.")
def lfit(input_file, output_file, figpath, txtpath, lifetimes):
    """Produce local fits of a dataset.
    """
    with h5py.File(input_file, "r") as infile:
        try:
            infile["average"]
        except KeyError:
            click.echo("This command only works with averaged data.")
            return
        fit_results = compute.local_fits(infile, lifetimes)
    with Path(output_file).open("w") as outfile:
        compute.save_lfit_params_as_txt(fit_results, outfile)
    return


cli.add_command(assemble)
cli.add_command(da)
cli.add_command(cd)
cli.add_command(inspect)
cli.add_command(shotslice)
cli.add_command(wlslice)
cli.add_command(absslice)
cli.add_command(lfit)