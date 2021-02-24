import click
import numpy as np
from . import core


def save_avg_as_txt(f, outdir, ds_name="average"):
    """Save the average dA for each wavelength as a CSV file.
    """
    da = f[ds_name]
    points, wls = da.shape
    ts = core.time_axis(length=points)
    outdata = np.empty((points, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving CSVs") as indices:
        for wl_idx in indices:
            outdata[:, 1] = da[:, wl_idx]
            outpath = outdir / f"{wavelengths[wl_idx]}.txt"
            core.save_txt(outdata, outpath)
    return


def save_avg_as_png(f, outdir, xlabel=None, ylabel=None, title=None):
    """Save the average dA for each wavelength as a PNG file.
    """
    da = f["average"]
    points, wls = da.shape
    ts = core.time_axis(length=points)
    outdata = np.empty((points, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving figures") as indices:
        for wl_idx in indices:
            outpath = outdir / f"{wavelengths[wl_idx]}.png"
            core.save_fig(ts, da[:, wl_idx], outpath, remove_dev=True)
    return


def save_avg_da_figures(f, outdir, ds_name="average"):
    """Save the average dA for each wavelength as a PNG file.
    """
    da = f[ds_name]
    points, wls = da.shape
    ts = core.time_axis(length=points)
    outdata = np.empty((points, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving figures") as indices:
        for wl_idx in indices:
            outpath = outdir / f"{wavelengths[wl_idx]}.png"
            core.save_fig(ts, da[:, wl_idx]*1_000, outpath, xlabel="Time (us)", ylabel="dA (mOD)",
                          title=f"{wavelengths[wl_idx]/100}nm", remove_dev=True)
    return


def save_avg_cd_figures(f, outdir, ds_name="average"):
    """Save the average dA for each wavelength as a PNG file.
    """
    cd = f[ds_name]
    points, wls = cd.shape
    ts = core.time_axis(length=points)
    outdata = np.empty((points, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving figures") as indices:
        for wl_idx in indices:
            outpath = outdir / f"{wavelengths[wl_idx]}.png"
            core.save_fig(ts, cd[:, wl_idx]*1_000, outpath, xlabel="Time (us)", ylabel="dCD",
                          title=f"{wavelengths[wl_idx]/100}nm", remove_dev=True)
    return


def save_lfit_params_as_txt(results, outfile):
    """Save the local fit amplitudes and lifetimes to a text file.
    """
    for wl in results.keys():
        outfile.write(f"[{wl:3d}]\n")
        count = 1
        for a, t in results[wl]:
            t_us = t * 1_000_000
            outfile.write(f"A{count}: {a:.2e}\n")
            outfile.write(f"T{count}: {t_us:.2f}us\n")
            count += 1
        outfile.write("\n")
    return


def save_da_shots_as_txt(outdir, ds, wl_idx):
    """Save each shot at a given wavelength as a CSV.
    """
    if not outdir.exists():
        outdir.mkdir()
    points, shots, wavelengths = ds.shape
    ts = core.time_axis(length=points)
    tmp = np.empty((points, shots, wavelengths))
    ds.read_direct(tmp)
    with click.progressbar(range(shots), label="Saving CSVs") as indices:
        for shot_idx in indices:
            save_data = np.empty((points, 2))
            save_data[:, 0] = ts
            save_data[:, 1] = tmp[:, shot_idx, wl_idx]
            filename = f"{shot_idx+1:03d}.txt"
            filepath = outdir / filename
            np.savetxt(filepath, save_data, delimiter=",")
    return


def save_raw_shots_as_txt(outdir, ds, wl_idx, chan, pump_idx):
    """Save each shot at a given wavelengths as a CSV.
    """
    if not outdir.exists():
        outdir.mkdir()
    points, _, shots, wavelengths, _ = ds.shape
    ts = core.time_axis(length=points)
    tmp = np.empty((points, 3, shots, wavelengths, 2))
    ds.read_direct(tmp)
    with click.progressbar(range(shots), label="Saving CSVs") as indices:
        for shot_idx in indices:
            save_data = np.empty((points, 2))
            save_data[:, 0] = ts
            save_data[:, 1] = tmp[:, chan.value, shot_idx, wl_idx, pump_idx]
            filename = f"{shot_idx+1:03d}.txt"
            filepath = outdir / filename
            np.savetxt(filepath, save_data, delimiter=",")
    return


def save_collapsed_as_txt(f, outdir):
    """Save collapsed data as CSV files.
    """
    all_data = f["collapsed"]
    ts = all_data[:, 0]
    data = all_data[:, 1:]
    points, wls = data.shape
    outdata = np.empty((points, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving CSVs") as indices:
        for wl_idx in indices:
            outdata[:, 1] = data[:, wl_idx]
            outpath = outdir / f"{wavelengths[wl_idx]}.txt"
            core.save_txt(outdata, outpath)
    return


def save_collapsed_as_png(f, outdir):
    """Save collapsed data as PNG files.
    """
    all_data = f["collapsed"]
    ts = all_data[:, 0]
    data = all_data[:, 1:]
    points, wls = data.shape
    outdata = np.empty((points, 2))
    outdata[:, 0] = ts
    wavelengths = f["wavelengths"]
    if not outdir.exists():
        outdir.mkdir()
    with click.progressbar(range(wls), label="Saving figures") as indices:
        for wl_idx in indices:
            outpath = outdir / f"{wavelengths[wl_idx]}.png"
            core.save_fig(ts, data[:, wl_idx], outpath, remove_dev=True)
    return


def make_import_script(filenames, output_file):
    """Generate a script that will import the specified files.
    """
    lines = []
    lines.append("print \"first spectrum for storage = \",?firstspec")
    lines.append("storespec = firstspec")
    lines.append("print \"\\nworking...\\n\"")
    for i in range(len(filenames)):
        original_filename = filenames[i]
        new_filename = str(original_filename.resolve()).replace("/", "\\")
        lines.append(f"open \"{new_filename}\" input 1")
        lines.append("spec0 = storespec")
        lines.append("len0 = 0")
        lines.append("print #1,?spec0 \"xy\"")
        lines.append("close 1")
        lines.append(f"comment0$ = \"{original_filename.stem}\"")
        lines.append("storespec = storespec + 1")
    contents = "\r\n".join(lines) + "\r\n"
    with output_file.open("w") as file:
        file.write(contents)
    return
