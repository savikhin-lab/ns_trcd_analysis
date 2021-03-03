import click
import h5py
import numpy as np
from pathlib import Path
from . import core
from . import compute
from . import extract
from . import gfit
from . import images
from . import noise
from . import raw2hdf5
from . import slices
from .core import Channels, valid_channel, load_dir_into_arr


POINTS = 20_000


@click.group()
def cli():
    pass


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the raw experiment data files.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file path at which to store the assembled experiment data.")
@click.option("-d", "--dark-signals", "dark_signals_file", default=None, type=click.Path(file_okay=True, dir_okay=False, exists=True), help="The file that contains the dark signals for each shot.")
def assemble(input_dir, output_file, dark_signals_file):
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
    if dark_signals_file is not None:
        dark_signals_file = Path(dark_signals_file)
    raw2hdf5.ingest(in_dir, outfile, dark_signals_file=dark_signals_file)


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
@click.option("-w", "--wavelength", type=click.INT, help="The wavelength to inspect.")
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
            wl_idx = core.index_for_wavelength(list(infile["wavelengths"]), wavelength)
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
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to subtract oscillations from.")
@click.option("-t", "--txt-data", "txt", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The oscillation data in text format.")
def rmosc(input_file, txt):
    """Remove oscillations from averaged dA or dCD data.
    """
    with h5py.File(input_file, "r+") as infile:
        try:
            avg_data = infile["average"]
        except KeyError:
            click.echo("File does not contain averaged dA or dCD data.")
            return
        osc_data = np.loadtxt(Path(txt), delimiter=",")[:, 1]
        if len(avg_data[:, 0]) != len(osc_data):
            click.echo("Averaged data and oscillation data are not the same length.")
            return
        num_wls = avg_data.shape[1]
        infile.copy(infile["average"], "osc_free")
        for i in range(num_wls):
            infile["osc_free"][:, i] -= osc_data
    return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to read from.")
@click.option("-p", "--points", default=1500, help="The number of points to use to calculate the offset (taken from the beginning of the curve.")
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
    contents = gfit.global_fit_file(task_names, lifetimes, amplitudes, input_spec, output_spec, instr_spec)
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
@click.option("-t", "--time-shift", required=True, type=click.FLOAT, help="The time in seconds by which to shift the time.")
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
@click.option("-t", "--cutoff-time", "times", required=True, multiple=True, type=click.FLOAT, help="The times at which to change the number of points to collapse.")
@click.option("-c", "--chunk-size", "cpoints", required=True, multiple=True, type=click.INT, help="The number of points to collapse at each interval.")
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
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to perform noise rejection on.")
@click.option("-s", "--sigmas", required=True, type=click.FLOAT, help="The number of std. devs. to use as a threshold for noise rejection.")
def noiserep(input_file, sigmas):
    """List the curves that would be rejected using the specified criteria.

    Note: This only works with dA or dCD files.
    """
    with h5py.File(input_file, "r") as infile:
        report = noise.reject_sigma(infile, sigmas)
        click.echo(report)
    return


@click.command()
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="The dA or dCD file to perform noise rejection on.")
@click.option("-o", "--output-file", required=True, type=click.Path(file_okay=True, dir_okay=False), help="The file to store the noise-rejected data in.")
@click.option("-s", "--sigmas", required=True, type=click.FLOAT, help="The number of std. devs. to use as a threshold for noise rejection.")
def noise_avg(input_file, output_file, sigmas):
    """Average dA or dCD data without including noisy shots.
    """
    with h5py.File(input_file, "r") as infile:
        report = noise.reject_sigma(infile, sigmas)
        noise.selective_average(infile, output_file, report)
    return


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the files to fit.")
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, dir_okay=True), help="The directory to store the fit results in.")
@click.option("--save-gfit-curves", is_flag=True, help="Save the fitted curves from the global fit.")
@click.option("--save-lfit-curves", is_flag=True, help="Save the fitted curves from the initial local fit.")
@click.option("-l", "--lifetime", "lifetimes", multiple=True, required=True, type=(click.FLOAT, click.FLOAT, click.FLOAT), help="A lifetime and the bounds within which it can vary entered as 'lower_bound, lifetime, upper_bound'. Pass one of these flags for each lifetime.")
@click.option("-a", "--fit-after", default=0, type=click.FLOAT, help="Only fit data after a certain time (useful to avoid pump spike).")
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
        fitted = compute.curves_from_fit(lfit_amps, [b.lifetime for b in bounded_lifetimes], ts, fit_after)
        compute.save_fitted_curves(output_dir / "lfit_curves", fitted, ts, wls)
    gfit_amps, gfit_lifetimes = compute.global_fit(data, ts, fit_after, lfit_amps, bounded_lifetimes)
    if save_gfit_curves:
        fitted = compute.curves_from_fit(gfit_amps, gfit_lifetimes, ts, fit_after)
        compute.save_fitted_curves(output_dir / "gfit_curves", fitted, ts, wls)
    compute.save_global_fit_spectra(output_dir, gfit_amps, wls, gfit_lifetimes)
    return


@click.command()
@click.option("-d", "--da-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the dA files to fit.")
@click.option("-c", "--cd-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="The directory containing the dCD files to fit.")
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, dir_okay=True), help="The directory in which to store the fit results.")
@click.option("-a", "--fit-after", default=0, type=click.FLOAT, help="Only fit data after this time (useful to avoid fitting scattered pump light).")
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
    if save_lfit_curves:
        lfit_curves = compute.curves_from_fit(lfit_amps, [b.lifetime for b in bounded_lifetimes], ts, fit_after)
        compute.save_double_lfits(output_dir, lfit_curves, ts, da_wls, cd_wls)
    gfit_amps, gfit_lifetimes = compute.global_fit(combined_data, ts, fit_after, lfit_amps, bounded_lifetimes)
    if save_gfit_curves:
        gfit_curves = compute.curves_from_fit(gfit_amps, gfit_lifetimes, ts, fit_after)
        compute.save_double_gfits(output_dir, gfit_curves, ts, da_wls, cd_wls)
    compute.save_double_fit_spectra(output_dir, gfit_amps, gfit_lifetimes, da_wls, cd_wls)


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
cli.add_command(noiserep)
cli.add_command(noise_avg)
cli.add_command(global_fit)
cli.add_command(double_fit)
