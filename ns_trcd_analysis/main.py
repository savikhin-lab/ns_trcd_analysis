import click
import h5py
import numpy as np
import csv
import json
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from . import core
from . import compute
from . import extract
from . import ssolve_gfit
from . import images
from . import noise
from . import raw2hdf5
from . import slices
from . import veusz
from .core import Channels, valid_channel, load_dir_into_arr


POINTS = 20_000


@click.group()
def cli():
    pass


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the raw experiment data files.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file path at which to store the assembled experiment data.")
@click.option("-d", "--dark-signals", "dark_signals_file", default=None, type=click.Path(file_okay=True, dir_okay=False, exists=True), help="The file that contains the dark signals for each shot.")
@click.option("--dark-par", default=None, type=click.FLOAT, help="The dark signal for the parallel channel.")
@click.option("--dark-perp", default=None, type=click.FLOAT, help="The dark signal for the perpendicular channel.")
@click.option("--dark-ref", default=None, type=click.FLOAT, help="The dark signal for the reference channel.")
def assemble(input_dir, output_file, dark_signals_file, dark_par, dark_perp, dark_ref):
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
    outfile = Path(output_file)
    if dark_signals_file:
        dark_signals_file = Path(dark_signals_file)
    dark_channels_count = sum([x is not None for x in (dark_par, dark_perp, dark_ref)])
    if dark_channels_count not in [0, 3]:
        click.echo("An incomplete set of dark signals was supplied.", err=True)
        return
    if dark_signals_file and (dark_channels_count == 3):
        click.echo("Please only supply a dark signals file or dark signals for each channel.", err=True)
        return
    raw2hdf5.ingest(in_dir, outfile, dark_signals_file=dark_signals_file,
                    dark_par=dark_par, dark_perp=dark_perp, dark_ref=dark_ref)


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
    click.echo("Loading file...")
    with h5py.File(output_file, "w") as outfile:
        with h5py.File(input_file, "r") as infile:
            (points, _, shots, wavelengths, pump_states) = infile["data"].shape
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
                    extract.save_avg_as_txt(outfile, Path(txt))
                if fig:
                    extract.save_avg_da_figures(outfile, Path(fig))
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
    click.echo("Loading file...")
    with h5py.File(output_file, "w") as outfile:
        with h5py.File(input_file, "r") as infile:
            (points, _, shots, wavelengths, pump_states) = infile["data"].shape
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
                    extract.save_avg_as_txt(outfile, Path(txt))
                if fig:
                    extract.save_avg_cd_figures(outfile, Path(fig))
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
@click.option("-w", "--wavelength", type=click.FLOAT, help="The wavelength to inspect.")
@click.option("--without-pump", is_flag=True, help="Extract images/CSVs from without-pump data.")
@click.option("--averaged", is_flag=True, help="Extract only averaged data if it exists.")
@click.option("--osc-free", is_flag=True, help="Extract only oscillation-free data if it exists.")
@click.option("--collapsed", is_flag=True, help="Extract only collapsed data if it exists.")
def export(input_file, fig, txt, format, channel, wavelength, without_pump, averaged, osc_free, collapsed):
    """Export data as CSVs or images.

    This works for both dA and raw data files (specified with the '-d' flag).
    """
    if (not fig) and (not txt):
        click.echo("Please select an output format with the '-f' and '-t' options.")
        return
    data_options = [averaged, osc_free, collapsed]
    if data_options.count(True) > 1:
        click.echo("Please choose at most one of '--averaged', '--osc-free', or '--collapsed'.")
        return
    with h5py.File(input_file, "r") as infile:
        if averaged:
            try:
                _ = infile["average"]
            except KeyError:
                click.echo("File does not contain averaged dA or dCD data.")
                return
            if txt:
                extract.save_avg_as_txt(infile, Path(txt))
            if fig:
                extract.save_avg_da_figures(infile, Path(fig))
            return
        elif osc_free:
            try:
                _ = infile["osc_free"]
            except KeyError:
                click.echo("File does not contain oscillation-free dA or dCD data.")
                return
            if txt:
                extract.save_avg_as_txt(infile, Path(txt), ds_name="osc_free")
            if fig:
                extract.save_avg_da_figures(infile, Path(fig), ds_name="osc_free")
            return
        elif collapsed:
            try:
                _ = infile["collapsed"]
            except KeyError:
                click.echo("File does not contain collapsed data.")
                return
            if txt:
                extract.save_collapsed_as_txt(infile, Path(txt))
            if fig:
                extract.save_collapsed_as_png(infile, Path(fig))
            return
        else:
            dataset = infile["data"]
            if not wavelength:
                click.echo("Please choose a wavelength.")
                return
            wl_idx = core.index_for_wavelength(list(infile["wavelengths"]), int(wavelength * 100))
            if wl_idx is None:
                click.echo("Wavelength not found.")
                return
            if format == "da":
                if fig:
                    images.dump_da_images(Path(fig), dataset, wl_idx)
                if txt:
                    extract.save_da_shots_as_txt(Path(txt), dataset, wl_idx)
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
                    extract.save_raw_shots_as_txt(Path(txt), dataset, wl_idx, chan, pump_idx)
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
            extract.save_avg_as_txt(file, Path(txt))
        if fig:
            extract.save_avg_da_figures(file, Path(fig))
    return


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the dCD data to subtract oscillations from.")
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, dir_okay=True), help="The directory to store the oscillation-free data in.")
@click.option("-a", "--after", default=1, show_default=True, type=click.FLOAT, help="Only fit the oscillations after this time.")
@click.option("-w", "--subtract-whole-curve", "whole_curve", is_flag=True, help="Subtract the whole oscillation curve after fitting the oscillations. The default behavior (without this flag) is to only subtract the oscillations after the time specified by the '-a' flag.")
def rmosc(input_dir, output_dir, after, whole_curve):
    """Remove oscillations from averaged dCD data.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    files = sorted([f for f in input_dir.iterdir() if f.suffix == ".txt"])
    if "85000" not in [f.stem for f in files]:
        click.echo("Data does not contain an 850nm curve.", err=True)
        return
    ts = np.loadtxt(files[0], delimiter=",")[:, 0]
    wavelengths = [int(f.stem) for f in files]
    osc_index = wavelengths.index(85000)
    osc_raw = np.loadtxt(files[osc_index], delimiter=",")[:, 1]
    osc_smoothed = osc_raw
    osc_smoothed[ts > after] = savgol_filter(osc_raw[ts > after], 11, 3)
    ts = core.time_axis()
    with click.progressbar(files, label="Removing oscillations") as files_iter:
        for i, f in enumerate(files_iter):
            original = np.loadtxt(f, delimiter=",")[:, 1]

            def minimize_me(x):
                return np.std(original[ts > after] - x * osc_smoothed[ts > after])

            res = minimize_scalar(minimize_me)
            scaled_osc = osc_smoothed
            scaled_osc[ts > after] = res.x * osc_smoothed[ts > after]
            if not whole_curve:
                scaled_osc[ts <= after] *= 0
            osc_free = original - scaled_osc
            out_data = np.empty((len(ts), 2))
            out_data[:, 0] = ts
            out_data[:, 1] = osc_free
            output_file = output_dir / f.name
            np.savetxt(output_file, out_data, delimiter=",")
    return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to read from.")
@click.option("-p", "--points", default=1500, show_default=True, help="The number of points to use to calculate the offset (taken from the beginning of the curve.")
@click.option("--each", is_flag=True, help="Remove the offset of each dA or dCD shot.")
@click.option("--average", is_flag=True, help="Remove the offset of the averaged dA or dCD data.")
@click.option("--osc-free", is_flag=True, help="Remove the offset of the oscillation-free dA or dCD data.")
@click.option("--collapsed", is_flag=True, help="Remove the offset of the collapsed dA or dCD data.")
def rmoffset(input_file, points, each, average, osc_free, collapsed):
    """Shift curves up or down such that the values before the pump are centered on zero.
    """
    with h5py.File(input_file, "r+") as file:
        if len(file["data"].shape) != 3:
            click.echo("File does not contain valid dA or dCD data (wrong dimensions).")
            return
        if each:
            compute.remove_da_shot_offsets(file["data"], points)
        if average:
            try:
                file["average"]
            except KeyError:
                click.echo("File does not contain averaged data.")
                return
            compute.remove_avg_offsets(file["average"], points)
        if osc_free:
            try:
                file["osc_free"]
            except KeyError:
                click.echo("File does not contain oscillation-free data.")
                return
            compute.remove_avg_offsets(file["osc_free"], points, ds_name="osc_free")
        if collapsed:
            try:
                file["collapsed"]
            except KeyError:
                click.echo("File does not contain collaped data.")
                return
            compute.remove_avg_offsets(file["collapsed"], points, ds_name="collapsed")
    return


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory that holds the original data files (used for task names).")
@click.option("-o", "--output-file", "output_file", type=click.Path(exists=False, file_okay=True, dir_okay=False), help="The filename of the generated fit file.")
@click.option("-l", "--lifetime", "lifetimes", multiple=True, required=True, type=click.FLOAT, help="The initial guesses for each lifetime. Multiple instances of this option are allowed.")
@click.option("--input-spec", required=True, type=click.INT, help="The first spectrum to read from.")
@click.option("--output-spec", required=True, type=click.INT, help="The first spectrum to write to.")
@click.option("--instr-spec", required=True, type=click.INT, help="The spectrum that holds the instrument function.")
def gfitfile(input_dir, output_file, lifetimes, input_spec, output_spec, instr_spec):
    indir = Path(input_dir)
    task_names = [f.stem for f in indir.iterdir() if f.suffix == ".txt"]
    task_names = sorted(task_names)
    amplitudes = [1 for _ in range(len(lifetimes))]
    outfile = Path(output_file)
    contents = ssolve_gfit.global_fit_file(task_names, lifetimes, amplitudes, input_spec, output_spec, instr_spec)
    with outfile.open("w") as file:
        file.write(contents)
    return


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory that holds the data files.")
@click.option("-o", "--output-file", "output_file", type=click.Path(exists=False, file_okay=True, dir_okay=False), help="The filename of the generated script.")
def importscript(input_dir, output_file):
    """Generate a script that imports the files in the specified directory.

    When run, the script will ask the user for the first spectrum in which to store the data.
    """
    input_dir = Path(input_dir)
    outfile = Path(output_file)
    files = sorted([f for f in input_dir.iterdir() if f.suffix == ".txt"])
    if len(files) == 0:
        click.echo("No valid files found in specified directory.")
        return
    extract.make_import_script(files, outfile)
    return


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory that holds the data files to shift.")
@click.option("-t", "--time-shift", required=True, type=click.FLOAT, help="The time in microseconds to add to every point on the time axis.")
def tshift(input_dir, time_shift):
    """Shift the time axis of data files in the specified directory.
    """
    input_dir = Path(input_dir)
    files = [f for f in input_dir.iterdir() if f.suffix == ".txt"]
    if len(files) == 0:
        click.echo("No valid files found in specified directory.")
        return
    for f in files:
        data = np.loadtxt(f, delimiter=",")
        data[:, 0] += time_shift
        np.savetxt(f, data, delimiter=",")
    return


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory that contains the data to collapse.")
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, dir_okay=True), help="The directory to put the collapsed data into.")
@click.option("-t", "--cutoff-time", "times", required=True, multiple=True, type=click.FLOAT, help="The times at which to change the number of points to collapse. You may specify this multiple times to collapse differently at different times.")
@click.option("-c", "--chunk-size", "cpoints", required=True, multiple=True, type=click.INT, help="The number of points to collapse at each interval. You may specify this multiple times to collapse differently at different times.")
def collapse(input_dir, output_dir, times, cpoints):
    """Collapse the data in the specified directory so that later times use fewer points.
    """
    if len(times) != len(cpoints):
        click.echo("There must be as many cutoff times as there are chunk sizes.")
        return
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = [f for f in input_dir.iterdir() if f.suffix == ".txt"]
    filenames = [f.name for f in files]
    ts = core.time_axis()
    for t in times:
        if t < ts[0]:
            click.echo(f"Time {t} occurs before the first point.")
            return
        if t > ts[-1]:
            click.echo(f"Time {t} occurs after the last point.")
            return
    num_points = core.POINTS
    num_wls = len(files)
    data_with_time = np.empty((num_points, num_wls + 1))
    data_with_time[:, 0] = ts
    for i, f in enumerate(files, start=1):
        data = np.loadtxt(f, delimiter=",")[:, 1]
        data_with_time[:, i] = data
    collapsed_data = compute.collapse(data_with_time, times, cpoints)
    collapsed_time = collapsed_data[:, 0]
    output_dir.mkdir(exist_ok=True)
    for i, f in enumerate(filenames, start=1):
        data = np.empty((len(collapsed_time), 2))
        data[:, 0] = collapsed_time
        data[:, 1] = collapsed_data[:, i]
        output_file = output_dir / f
        np.savetxt(output_file, data, delimiter=",")
    return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The raw or dA data file to read from.")
@click.option("-d", "--data-format", type=click.Choice(["raw", "da"]), required=True, help="The format of the data file.")
@click.option("-c", "--channel", type=click.Choice(["par", "perp", "ref"]), help="If the format of the data is 'raw', which channel to slice.")
@click.option("-f", "--figure-path", "figpath", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txtpath", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("--slice-time", "stime", type=click.FLOAT, help="Select the slice closest to the specified time (in us).")
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
            t = core.time_axis()[s_idx]
            core.save_fig(shots, s, figpath, xlabel="Shot Number", title=f"{wavelength}nm, t={t:.2f}us")
    return


@click.command()
@click.option("-i", "--input-file", type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA data file to read from.")
@click.option("-d", "--input-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the files to read from.")
@click.option("-f", "--figure-path", "fig", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txt", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("--slice-time", "stime", type=click.FLOAT, help="Select the slice closest to the specified time (in us).")
@click.option("--slice-index", "sindex", type=click.INT, help="Select the slice at the specified index along the time axis.")
@click.option("--averaged", is_flag=True, help="Take the slice from averaged data.")
@click.option("--osc-free", is_flag=True, help="Take the slice from oscillation-free data.")
@click.option("--collapsed", is_flag=True, help="Take the slice from collapsed data.")
def wlslice(input_file, input_dir, fig, txt, stime, sindex, averaged, osc_free, collapsed):
    """Create a dA or dCD slice at all wavelengths for a specified time.

    Note: This command is only valid for averaged data.
    """
    if (txt is None) and (fig is None):
        click.echo("No output has been chosen. See '-f' or '-t'.", err=True)
        return
    if input_file:
        data_options = [averaged, osc_free, collapsed]
        if data_options.count(True) != 1:
            click.echo("Choose a data source using '--averaged', '--osc-free', or '--collapsed'.")
            return
        with h5py.File(input_file, "r") as infile:
            if averaged:
                try:
                    data = infile["average"]
                except KeyError:
                    click.echo("File does not contain averaged data.")
                    return
            elif osc_free:
                try:
                    data = infile["osc_free"]
                except KeyError:
                    click.echo("File does not contain oscillation-free data.")
                    return
            elif collapsed:
                try:
                    data = infile["collapsed"]
                except KeyError:
                    click.echo("File does not contain collapsed data.")
                    return
            wavelengths = [x / 100 for x in infile["wavelengths"]]
            points = data.shape[0]
    elif input_dir:
        input_dir = Path(input_dir)
        files = [f for f in input_dir.iterdir() if f.suffix == ".txt"]
        first_file = np.loadtxt(files[0], delimiter=",")
        points = first_file.shape[0]
        wavelengths = [int(f.stem) for f in files]
        data = np.empty((points, len(files)))
        for i, f in enumerate(files):
            data[:, i] = np.loadtxt(f, delimiter=",")[:, 1]
    else:
        click.echo("Choose an input source with --input-file or --input-dir", err=True)
        return
    if not slices.valid_shot_slice_point(stime, sindex, points):
        return
    if sindex is None:
        if collapsed:
            ts = data[:, 0]
        else:
            ts = core.time_axis()
        s_idx = slices.index_nearest_to_value(ts, stime)
        if s_idx is None:
            click.echo("Slice time is out of range.")
            return
    else:
        s_idx = sindex
    if collapsed:
        slice_data = data[s_idx, 1:]
    else:
        slice_data = data[s_idx, :]
    if txt:
        txtdata = np.empty((len(wavelengths), 2))
        txtdata[:, 0] = wavelengths
        txtdata[:, 1] = slice_data
        core.save_txt(txtdata, Path(txt))
    if fig:
        t = ts[s_idx]
        core.save_fig(wavelengths, slice_data * 1_000, fig, xlabel="Wavelength",
                      ylabel="dA (mOD)", title=f"Slice at t={t:.2f}us")
        return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The raw data file to read from.")
@click.option("-f", "--figure-path", "figpath", type=click.Path(file_okay=True, dir_okay=False), help="Generate a figure at the specified path.")
@click.option("-t", "--txt-path", "txtpath", type=click.Path(file_okay=True, dir_okay=False), help="Save a CSV file at the specified path.")
@click.option("--slice-time", "stime", type=click.FLOAT, help="Select the slice closest to the specified time (in us).")
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
            t = core.time_axis()[s_idx]
            core.save_fig(shots, s, figpath, xlabel="Shot number", ylabel="Abs.",
                          title=f"Slice at {wavelength}nm, t={t:.2f}us")
        return


@click.command()
@click.option("-d", "--data-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to perform noise rejection on.")
@click.option("-f", "--filter-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file to store the list of rejected shots in.")
@click.option("-s", "--scale", default=2.0, show_default=True, type=click.FLOAT, help="The number of std. devs. to use as a threshold for noise rejection.")
def sigma_filter(data_file, filter_file, scale):
    """Reject shots based on whether their noise is within a certain number of standard deviations of the mean.
    """
    data_file = Path(data_file)
    filter_file = Path(filter_file)
    with h5py.File(data_file, "r") as f:
        data = np.empty_like(f["data"])
        f["data"].read_direct(data, np.s_[:, :, :], np.s_[:, :, :])
    filtered = noise.reject_sigma(data, scale)
    if filter_file.exists():
        old_filtered = noise.load_filter_list(filter_file)
        filtered = noise.merge_filter_lists(filtered, old_filtered)
    with filter_file.open("w") as f:
        json.dump(filtered, f)


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the files to fit.")
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, dir_okay=True), help="The directory to store the fit results in.")
@click.option("--save-gfit-curves", is_flag=True, help="Save the fitted curves from the global fit.")
@click.option("--save-lfit-curves", is_flag=True, help="Save the fitted curves from the initial local fit.")
@click.option("-l", "--lifetime", "lifetimes", multiple=True, required=True, type=(click.FLOAT, click.FLOAT, click.FLOAT), help="A lifetime and the bounds within which it can vary entered as 'lower_bound, lifetime, upper_bound'. Pass one of these flags for each lifetime.")
@click.option("-a", "--fit-after", default=0, show_default=True, type=click.FLOAT, help="Only fit data after a certain time (useful to avoid pump spike).")
def global_fit(input_dir, output_dir, save_gfit_curves, save_lfit_curves, lifetimes, fit_after):
    """Do a global fit with the provided lifetimes.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    bounded_lifetimes = compute.bounded_lifetimes_from_args(lifetimes)
    data, ts = load_dir_into_arr(input_dir)
    wls = [int(f.stem) for f in sorted(input_dir.iterdir()) if f.suffix == ".txt"]
    lfit_amps = compute.lfits_for_gfit(data, ts, fit_after, bounded_lifetimes)
    if save_lfit_curves:
        fitted = compute.curves_from_fit(lfit_amps, [b.lifetime for b in bounded_lifetimes], ts)
        compute.save_fitted_curves(output_dir / "lfit_curves", fitted, ts, wls)
    gfit_amps, gfit_lifetimes = compute.global_fit(data, ts, fit_after, lfit_amps, bounded_lifetimes)
    if save_gfit_curves:
        fitted = compute.curves_from_fit(gfit_amps, gfit_lifetimes, ts)
        compute.save_fitted_curves(output_dir / "gfit_curves", fitted, ts, wls)
    compute.save_global_fit_spectra(output_dir, gfit_amps, wls, gfit_lifetimes)
    return


@click.command()
@click.option("-d", "--da-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the dA files to fit.")
@click.option("-c", "--cd-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the dCD files to fit.")
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, dir_okay=True), help="The directory in which to store the fit results.")
@click.option("-a", "--fit-after", default=0, show_default=True, type=click.FLOAT, help="Only fit data after this time (useful to avoid fitting scattered pump light).")
@click.option("-l", "--lifetime", "lifetimes", multiple=True, required=True, type=(click.FLOAT, click.FLOAT, click.FLOAT), help="A lifetime and the bounds within which it can vary entered as 'lower_bound, lifetime, upper_bound'. Pass one of these flags for each lifetime.")
@click.option("--save-gfit-curves", is_flag=True, help="Save the fitted curves from the global fit.")
@click.option("--save-lfit-curves", is_flag=True, help="Save the fitted curves from the initial local fit.")
def double_fit(da_dir, cd_dir, output_dir, fit_after, lifetimes, save_gfit_curves, save_lfit_curves):
    """Do a global fit of the dA and dCD data at the same time so that they share lifetimes.
    """
    da_dir = Path(da_dir)
    cd_dir = Path(cd_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    bounded_lifetimes = compute.bounded_lifetimes_from_args(lifetimes)
    da_data, ts = load_dir_into_arr(da_dir)
    da_wls = [int(f.stem) for f in sorted(da_dir.iterdir()) if f.suffix == ".txt"]
    cd_data, _ = load_dir_into_arr(cd_dir)
    cd_wls = [int(f.stem) for f in sorted(cd_dir.iterdir()) if f.suffix == ".txt"]
    combined_data = np.hstack((da_data, cd_data))
    lfit_amps = compute.lfits_for_gfit(combined_data, ts, fit_after, bounded_lifetimes)
    gfit_amps, gfit_lifetimes = compute.global_fit(combined_data, ts, fit_after, lfit_amps, bounded_lifetimes)
    if save_lfit_curves:
        lfit_curves = compute.curves_from_fit(lfit_amps, [b.lifetime for b in bounded_lifetimes], ts)
        compute.save_double_lfits(output_dir, lfit_curves, ts, da_wls, cd_wls)
    if save_gfit_curves:
        gfit_curves = compute.curves_from_fit(gfit_amps, gfit_lifetimes, ts)
        compute.save_double_gfits(output_dir, gfit_curves, ts, da_wls, cd_wls)
    compute.save_double_fit_spectra(output_dir, gfit_amps, gfit_lifetimes, da_wls, cd_wls)


@click.command()
@click.option("-d", "--da-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the dA files to fit.")
@click.option("-c", "--cd-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the dCD files to fit.")
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, dir_okay=True), help="The directory in which to store the fit results.")
@click.option("-a", "--fit-after", default=0, show_default=True, type=click.FLOAT, help="Only fit data after this time (useful to avoid fitting scattered pump light).")
@click.option("-l", "--lifetime", "lifetimes", multiple=True, required=True, type=(click.FLOAT, click.FLOAT, click.FLOAT), help="A lifetime and the bounds within which it can vary entered as 'lower_bound, lifetime, upper_bound'. Pass one of these flags for each lifetime.")
@click.option("--save-gfit-curves", is_flag=True, help="Save the fitted curves from the global fit.")
@click.option("--save-lfit-curves", is_flag=True, help="Save the fitted curves from the initial local fit.")
def fixed_double_fit(da_dir, cd_dir, output_dir, fit_after, lifetimes, save_gfit_curves, save_lfit_curves):
    """Do a global fit of dA and dCD where the lifetimes come from just the dA data.
    """
    da_dir = Path(da_dir)
    cd_dir = Path(cd_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    bounded_lifetimes = compute.bounded_lifetimes_from_args(lifetimes)
    da_data, ts = load_dir_into_arr(da_dir)
    da_wls = [int(f.stem) for f in sorted(da_dir.iterdir()) if f.suffix == ".txt"]
    cd_data, _ = load_dir_into_arr(cd_dir)
    cd_wls = [int(f.stem) for f in sorted(cd_dir.iterdir()) if f.suffix == ".txt"]
    da_lfit_amps = compute.lfits_for_gfit(da_data, ts, fit_after, bounded_lifetimes)
    cd_lfit_amps = compute.lfits_for_gfit(cd_data, ts, fit_after, bounded_lifetimes)
    da_gfit_amps, gfit_lifetimes = compute.global_fit(da_data, ts, fit_after, da_lfit_amps, bounded_lifetimes)
    cd_gfit_amps = compute.fixed_lifetime_global_fit(cd_data, ts, fit_after, cd_lfit_amps, gfit_lifetimes)
    compute.save_global_fit_spectra(output_dir / "da_spectra", da_gfit_amps, da_wls, gfit_lifetimes)
    compute.save_global_fit_spectra(output_dir / "cd_spectra", cd_gfit_amps, cd_wls, gfit_lifetimes)
    if save_lfit_curves:
        da_curves = compute.curves_from_fit(da_lfit_amps, gfit_lifetimes, ts)
        cd_curves = compute.curves_from_fit(cd_lfit_amps, gfit_lifetimes, ts)
        compute.save_fitted_curves(output_dir / "da_lfit_curves", da_curves, ts, da_wls)
        compute.save_fitted_curves(output_dir / "cd_lfit_curves", cd_curves, ts, cd_wls)
    if save_gfit_curves:
        da_curves = compute.curves_from_fit(da_gfit_amps, gfit_lifetimes, ts)
        cd_curves = compute.curves_from_fit(cd_gfit_amps, gfit_lifetimes, ts)
        compute.save_fitted_curves(output_dir / "da_gfit_curves", da_curves, ts, da_wls)
        compute.save_fitted_curves(output_dir / "cd_gfit_curves", cd_curves, ts, cd_wls)


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory of text files to assemble into an NPY file.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The new file to store the data in.")
def txtdir2npy(input_dir, output_file):
    """Load a directory of CSV files into a single NPY file.

    \b
    One copy of the first column in the text files (time, wavelength, etc) will be included in the
    first column of the NPY file.
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    data, xs = load_dir_into_arr(input_dir)
    xs = xs.reshape((len(xs), 1))
    out_data = np.hstack((xs, data))
    np.save(output_file, out_data)


@click.command()
@click.option("-d", "--data-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to examine for noise rejection.")
@click.option("-f", "--filter-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file to store a list of rejected shots in. If this file exists, the contents are merged with the results of this filter.")
@click.option("-s", "--scale", default=1.25, show_default=True, type=click.FLOAT, help="The filter cutoff in terms of the mean of the integral of the band between the upper and lower frequencies.")
@click.option("--f-upper", default=0.8, show_default=True, type=click.FLOAT, help="The upper cutoff frequency in MHz.")
@click.option("--f-lower", default=0.2, show_default=True, type=click.FLOAT, help="The lower cutoff frequency in MHz.")
def fft_filter(data_file, filter_file, scale, f_upper, f_lower):
    """Produce a list of shots to filter based on the noise between an upper and lower frequency.

    \b
    The noise between the upper and lower frequencies is integrated and averaged for each wavelength.
    If the integrated noise for a shot is greater than 'scale' times the mean of the integrated noise
    for the wavelength, that shot is filtered out.

    The noise file is a JSON file where the top level keys correspond to wavelengths, and the values of
    those keys are arrays of shots to ignore when averaging the data.
    """
    data_file = Path(data_file)
    filter_file = Path(filter_file)
    with h5py.File(data_file, "r") as infile:
        data = np.empty_like(infile["data"])
        infile["data"].read_direct(data, np.s_[:, :, :], np.s_[:, :, :])
    filtered = noise.reject_fft(data, scale, f_upper, f_lower)
    if filter_file.exists():
        old_filtered = noise.load_filter_list(filter_file)
        filtered = noise.merge_filter_lists(filtered, old_filtered)
    with filter_file.open("w") as f:
        json.dump(filtered, f)


@click.command()
@click.option("-d", "--data-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to examine for noise rejection.")
@click.option("-f", "--filter-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file to store a list of rejected shots in. If this file exists, the contents are merged with the results of this filter.")
@click.option("-s", "--scale", default=0.5, show_default=True, type=click.FLOAT, help="Shots whose integral is below 'scale * mean' will be filtered.")
@click.option("--start", default=0.1, show_default=True, type=click.FLOAT, help="The start time for the integral.")
@click.option("--stop", default=50, show_default=True, type=click.FLOAT, help="The stop time for the integral.")
def int_filter(data_file, filter_file, scale, start, stop):
    """Produce a list of shots to filter based on their integral between a start and stop time.
    """
    data_file = Path(data_file)
    filter_file = Path(filter_file)
    with h5py.File(data_file, "r") as infile:
        data = np.empty_like(infile["data"])
        infile["data"].read_direct(data, np.s_[:, :, :], np.s_[:, :, :])
    filtered = noise.reject_integral(data, scale, start, stop)
    if filter_file.exists():
        old_filtered = noise.load_filter_list(filter_file)
        filtered = noise.merge_filter_lists(filtered, old_filtered)
    with filter_file.open("w") as f:
        json.dump(filtered, f)


@click.command()
@click.option("-d", "--data-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to examine for noise rejection.")
@click.option("-f", "--filter-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file to store a list of rejected shots in. If this file exists, the contents are merged with the results of this filter.")
@click.option("-t", "--threshold", default=1e-5, type=click.FLOAT, help="The threshold for when to stop throwing away bad shots.")
def incremental_filter(data_file, filter_file, threshold):
    data_file = Path(data_file)
    filter_file = Path(filter_file)
    with h5py.File(data_file, "r") as infile:
        data = np.empty_like(infile["data"])
        infile["data"].read_direct(data, np.s_[:, :, :], np.s_[:, :, :])
    if filter_file.exists():
        filtered = noise.load_filter_list(filter_file)
    else:
        filtered = {x: [] for x in range(data.shape[2])}
    filtered = noise.incremental_filter(data, filtered, threshold)
    with filter_file.open("w") as f:
        json.dump(filtered, f)


@click.command()
@click.option("-d", "--data-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to average.")
@click.option("-f", "--filter-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The file that contains the shots to exclude from the average.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file to store the noise-rejected average in.")
def filter_avg(data_file, filter_file, output_file):
    """Average data while excluding shots contained in the filter file.
    """
    data_file = Path(data_file)
    filter_file = Path(filter_file)
    output_file = Path(output_file)
    filter_list = noise.load_filter_list(filter_file)
    with h5py.File(data_file, "r") as infile:
        noise.selective_average(infile, output_file, filter_list)


@click.command()
@click.option("-r", "--raw-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory of dA or dCD that were fit.")
@click.option("-f", "--fit-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory of fitted curves.")
@click.option("-a", "--after", default=0.1, show_default=True, type=click.FLOAT, help="Only compare fits after a certain time.")
def chi2(raw_dir, fit_dir, after):
    """Calculate the chi2 of a global fit.
    """
    raw_dir = Path(raw_dir)
    fit_dir = Path(fit_dir)
    raw_data, t = load_dir_into_arr(raw_dir)
    fit_data, _ = load_dir_into_arr(fit_dir)
    diffs = raw_data[t > after, :] - fit_data[t > after, :]
    points = raw_data[t > after, :].shape[0] * raw_data.shape[1]
    norm = np.linalg.norm(diffs) / points
    click.echo(f"Chi2: {norm:.2e}")


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing files to fit.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The Veusz file to create.")
@click.option("--x-lower", type=click.FLOAT, help="A lower bound on the x-axis.")
@click.option("--x-upper", type=click.FLOAT, help="A upper bound on the x-axis.")
@click.option("--x-label", type=click.STRING, help="A name for the x-axis.")
@click.option("--y-label", type=click.STRING, help="A name for the y-axis.")
@click.option("--combined", is_flag=True, help="Put all the data on a single graph.")
def plot_dir(input_dir, output_file, x_lower, x_upper, x_label, y_label, combined):
    """Make plots from the files in a directory.

    All of the files must share the same x-axis.

    If you plan to do any operations on the datasets in Veusz, make sure there are no decimal places
    in the lifetimes. Veusz does not know how to parse expressions with decimal places in dataset names.
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    files = sorted([f for f in input_dir.iterdir() if f.suffix == ".txt"])
    options = {
        "x_lower": x_lower,
        "x_upper": x_upper,
        "x_label": x_label,
        "y_label": y_label
    }
    if combined:
        options["key"] = True
        veusz.plot_combined(output_file, files, options)
    else:
        veusz.plot_separate(output_file, files, options)


@click.command()
@click.option("-d", "--da-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the dA data used in the fit.")
@click.option("-s", "--spectra-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the spectra from the fit.")
@click.option("-f", "--fitted-curves-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the global fit curves from the fit.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The Veusz file to create.")
def plot_gfit(da_dir, spectra_dir, fitted_curves_dir, output_file):
    """Assemble the plots from a global fit.

    Compares the raw data to the fits, and plots the spectra together.

    If you plan to do any operations on the datasets in Veusz, make sure there are no decimal places
    in the lifetimes. Veusz does not know how to parse expressions with decimal places in dataset names."""
    da_dir = Path(da_dir)
    spectra_dir = Path(spectra_dir)
    fitted_curves_dir = Path(fitted_curves_dir)
    output_file = Path(output_file)
    raw_files = sorted([f for f in da_dir.iterdir() if f.suffix == ".txt"])
    curve_files = sorted([f for f in fitted_curves_dir.iterdir() if f.suffix == ".txt"])
    spectra_files = sorted([f for f in spectra_dir.iterdir() if f.suffix == ".txt"])
    veusz.plot_gfit(raw_files, curve_files, spectra_files, output_file)


@click.command()
@click.option("-d", "dirs", required=True, multiple=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directories containing the data to compare.")
@click.option("-l", "labels", multiple=True, type=click.STRING, help="The labels for the data from each directory.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The Veusz file to create.")
@click.option("--x-lower", type=click.FLOAT, help="A lower bound on the x-axis.")
@click.option("--x-upper", type=click.FLOAT, help="A upper bound on the x-axis.")
@click.option("--x-label", type=click.STRING, help="A name for the x-axis.")
@click.option("--y-label", type=click.STRING, help="A name for the y-axis.")
def plot_compared(dirs, labels, output_file, x_lower, x_upper, x_label, y_label):
    """Plot corresponding files from each directory together.

    The files from each directory are sorted and then all of the first files are plotted together,
    all the second files are plotted together, etc. If no labels are supplied, the filenames will be used as
    the plot names.

    If you plan to do any operations on the datasets in Veusz, make sure there are no decimal places
    in the lifetimes. Veusz does not know how to parse expressions with decimal places in dataset names."""
    dirs = [Path(d) for d in dirs]
    output_file = Path(output_file)
    options = {
        "x_lower": x_lower,
        "x_upper": x_upper,
        "x_label": x_label,
        "y_label": y_label
    }
    veusz.plot_compared(dirs, output_file, labels, options)


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the files to add to the filter.")
@click.option("-f", "--filter-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file to store a list of rejected shots in. If this file exists, the contents are merged with the results of this filter.")
@click.option("--index", required=True, type=click.INT, help="The wavelength index to add the shots to.")
def add_to_filter(input_dir, filter_file, index):
    """Add files in the target directory to the filter file.

    The names of files are used as the shot numbers (minus one to make them zero-indexed).
    """
    input_dir = Path(input_dir)
    filter_file = Path(filter_file)
    old_filtered = noise.load_filter_list(filter_file)
    shots = sorted([int(f.stem) - 1 for f in input_dir.iterdir() if f.suffix == ".png"])
    tmp_filtered = {index: shots}
    new_filtered = noise.merge_filter_lists(old_filtered, tmp_filtered)
    with filter_file.open("w") as f:
        json.dump(new_filtered, f)


@click.command()
@click.option("-d", "--data-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The file containing the dA or dCD data to be filtered.")
@click.option("-f", "--filter-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file to store a list of rejected shots in. If this file exists, the contents are merged with the results of this filter.")
@click.option("--fit-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the global fits.")
@click.option("-s", "--scale", default=1.0, type=click.FLOAT, help="The number of standard deviations that act as the threshold for filtering.")
def filter_from_fits(data_file, filter_file, fit_dir, scale):
    data_file = Path(data_file)
    filter_file = Path(filter_file)
    fit_dir = Path(fit_dir)
    with h5py.File(data_file, "r") as infile:
        data = np.empty_like(infile["data"])
        infile["data"].read_direct(data, np.s_[:, :, :], np.s_[:, :, :])
    fits, t = core.load_dir_into_arr(fit_dir)
    filtered = noise.filter_from_fits(data, fits, t, scale)
    if filter_file.exists():
        old_filtered = noise.load_filter_list(filter_file)
        filtered = noise.merge_filter_lists(filtered, old_filtered)
    with filter_file.open("w") as f:
        json.dump(filtered, f)


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The absorption spectrometer file to process.")
def clean_abs(input_file):
    """Format a CSV file from the UV-VIS absorption spectrometer."""
    input_file = Path(input_file)
    with input_file.open("r") as infile:
        reader = csv.reader(infile, delimiter=",")
        wls = []
        abs_data = []
        for i, row in enumerate(reader):
            if i in [0, 1]:
                continue
            try:
                wls.append(int(row[0]))
                abs_data.append(float(row[1]))
            except (ValueError, IndexError):
                break
        wls = reversed(wls)
        baseline = np.mean(abs_data[:15])
        abs_data = reversed(abs_data)
        abs_data = [x - baseline for x in abs_data]
    with input_file.open("w", newline="\n") as outfile:
        writer = csv.writer(outfile)
        for w, a in zip(wls, abs_data):
            writer.writerow([w, a])


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The oscilloscope output file to process.")
def clean_scope(input_file):
    """Format a CSV from the oscilloscope."""
    input_file = Path(input_file)
    rows = []
    with input_file.open("r") as infile:
        for row in csv.reader(infile, delimiter=","):
            rows.append(f"{row[3]},{row[4].strip()}")
    with input_file.open("w") as outfile:
        outfile.write("\n".join(rows))


cli.add_command(assemble)
cli.add_command(da)
cli.add_command(cd)
cli.add_command(export)
cli.add_command(shotslice)
cli.add_command(wlslice)
cli.add_command(absslice)
cli.add_command(split)
cli.add_command(average)
cli.add_command(rmosc)
cli.add_command(rmoffset)
cli.add_command(gfitfile)
cli.add_command(importscript)
cli.add_command(tshift)
cli.add_command(collapse)
cli.add_command(global_fit)
cli.add_command(double_fit)
cli.add_command(fixed_double_fit)
cli.add_command(txtdir2npy)
cli.add_command(fft_filter)
cli.add_command(sigma_filter)
cli.add_command(int_filter)
cli.add_command(filter_avg)
cli.add_command(chi2)
cli.add_command(plot_dir)
cli.add_command(plot_gfit)
cli.add_command(plot_compared)
cli.add_command(add_to_filter)
cli.add_command(filter_from_fits)
cli.add_command(clean_abs)
cli.add_command(clean_scope)
cli.add_command(incremental_filter)
