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
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the raw experiment data files.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file path at which to store the assembled experiment data.")
def assemble(input_dir, output_file):
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
    outfile = in_dir / output_file
    raw2hdf5.ingest(in_dir, outfile)


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The raw or dA data file to read from.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file path at which to store the results.")
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
            without_pump = (pump_states == 2)
            outfile.create_dataset("data", (points, shots, wavelengths))
            outfile.create_dataset("wavelengths", (wavelengths,), data=infile["wavelengths"])
            if perp:
                compute.compute_perp_da(infile, outfile)
            else:
                if without_pump:
                    compute.compute_da_with_and_without_pump(infile, outfile)
                else:
                    compute.compute_da_always_pumped(infile, outfile)
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
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The raw or dA data file to read from.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file path at which to store the results.")
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
            without_pump = (pump_states == 2)
            outfile.create_dataset("data", (points, shots, wavelengths))
            outfile.create_dataset("wavelengths", (wavelengths,), data=infile["wavelengths"])
            if without_pump:
                compute.compute_cd_with_and_without_pump(infile, outfile, delta)
            else:
                compute.compute_cd_always_pumped(infile, outfile, delta)
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
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The raw or dA data file to read from.")
@click.option("-f", "--figure-path", "fig", required=False, type=click.Path(exists=False, file_okay=False, dir_okay=True), help="The directory in which to store images of each shot.")
@click.option("-t", "--txt-path", "txt", required=False, type=click.Path(exists=False, file_okay=False, dir_okay=True), help="The directory in which to store CSVs of each shot.")
@click.option("-d", "--data-format", "format", type=click.Choice(["raw", "da"]), help="The format of the data file.")
@click.option("-c", "--channel", type=click.Choice(["par", "perp", "ref"]), help="If the format of the data is 'raw', which channel to inspect.")
@click.option("-w", "--wavelength", type=click.INT, required=True, help="The wavelength to inspect.")
@click.option("--without-pump", is_flag=True, help="Extract images/CSVs from without-pump data.")
@click.option("-a", "--average", is_flag=True, help="Extract only averaged data if it exists.")
def extract(input_file, fig, txt, format, channel, wavelength, without_pump, average):
    """Generate images of each shot in a data file.

    This works for both dA and raw data files (specified with the '-d' flag).
    """
    if (not fig) and (not txt):
        click.echo("Please select an output format with the -f/-t options.")
        return
    with h5py.File(input_file, "r") as infile:
        wl_idx = core.index_for_wavelength(list(infile["wavelengths"]), wavelength)
        if wl_idx is None:
            click.echo("Wavelength not found.")
            return
        if average:
            try:
                data = infile["average"][:, wl_idx]
            except KeyError:
                click.echo("File does not contain averaged dA or dCD data.")
                return
            ts = core.time_axis(length=len(data))
            if txt:
                txt_data = np.empty((len(data), 2))
                txt_data[:, 0] = ts
                txt_data[:, 1] = data
                core.save_txt(txt_data, Path(txt))
            if fig:
                core.save_fig(ts, data, Path(fig))
        else:
            dataset = infile["data"]
            if format == "da":
                if fig:
                    images.dump_da_images(Path(fig), dataset, wl_idx)
                if txt:
                    compute.save_da_shots_as_txt(Path(txt), dataset, wl_idx)
            elif format == "raw":
                pump_states = dataset.shape[4]
                if without_pump:
                    if pump_states > 1:
                        pump_idx = 1
                    else:
                        click.echo("Data file only contains with-pump data.")
                        return
                else:
                    pump_idx = 0
                if not channel:
                    click.echo("Raw data format requires a channel specifier. See the '-c' option.", err=True)
                    return
                chan = core.CHANNEL_MAP[channel]
                if fig:
                    images.dump_raw_images(Path(fig), chan, dataset, wl_idx, pump_idx)
                if txt:
                    compute.save_raw_shots_as_txt(Path(txt), dataset, wl_idx, chan, pump_idx)
            else:
                click.echo("Invalid data format", err=True)
                return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to split.")
@click.option("-s", "--size", required=True, type=click.INT, help="The number of shots in each split.")
def split(input_file, size):
    """Split a data file into chunks of a given size.

    This only works for dA and dCD files. If the total number of shots isn't a multiple of `size`, the last split
    will contain fewer shots.
    """
    input_file_path = Path(input_file)
    parent_path = input_file_path.parent
    input_file_stem = input_file_path.stem
    with h5py.File(input_file, "r") as infile:
        points, shots, wavelengths = infile["data"].shape
        original = np.empty((points, shots, wavelengths))
        infile["data"].read_direct(original)
        splits = core.compute_splits(shots, size)
        for i, (start, stop) in enumerate(splits):
            split_file = parent_path / (input_file_stem + f"_split{i}.h5")
            if split_file.exists():
                click.echo("A split file with a conflicting name already exists.")
                return
            with h5py.File(split_file, "w") as outfile:
                tmp_ds = original[:, start:stop, :]
                outfile.copy(infile["wavelengths"], "wavelengths")
                outfile.create_dataset("data", data=tmp_ds)
    return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to average.")
@click.option("-f", "--figure-path", "fig", type=click.Path(file_okay=False, dir_okay=True), help="Save a figure of the average dA. Only valid with the '-a' option.")
@click.option("-t", "--save-txt-path", "txt", type=click.Path(file_okay=False, dir_okay=True), help="Save a CSV of the average dA. Only valid with the '-a' option.")
def average(input_file, fig, txt):
    """Average the data contained in a dA or dCD file.
    """
    with h5py.File(input_file, "r+") as file:
        compute.average(file)
        if txt:
            compute.save_avg_as_txt(file, Path(txt))
        if fig:
            compute.save_da_figures(file, Path(fig))
    return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The raw or dA data file to read from.")
@click.option("-d", "--data-format", type=click.Choice(["raw", "da"]), required=True, help="The format of the data file.")
@click.option("-c", "--channel", type=click.Choice(["par", "perp", "ref"]), help="If the format of the data is 'raw', which channel to slice.")
@click.option("-f", "--figure-path", "figpath", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txtpath", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("--slice-time", "stime", type=click.FLOAT, help="Select the slice closest to the specified time.")
@click.option("--slice-index", "sindex", type=click.INT, help="Select the slice at the specified index along the time axis.")
@click.option("-w", "--wavelength", type=click.INT, required=True, help="The wavelength to create a slice of.")
def shotslice(input_file, data_format, channel, figpath, txtpath, stime, sindex, wavelength):
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
        if data_format == "raw":
            if not core.valid_channel(channel):
                return
            chan = core.CHANNEL_MAP[channel]
            s = infile["data"][s_idx, chan.value, :, wl_idx, 0]
        elif data_format == "da":
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
            core.save_fig(shots, s, figpath, xlabel="Shot Number", title=f"{wavelength}nm, t={t:.2f}us")
    return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA data file to read from.")
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
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The raw data file to read from.")
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
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The input file containing either dA or dCD data.")
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
cli.add_command(extract)
cli.add_command(shotslice)
cli.add_command(wlslice)
cli.add_command(absslice)
cli.add_command(lfit)
cli.add_command(split)
cli.add_command(average)