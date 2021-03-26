from itertools import chain


style_settings = """Set('colorTheme', u'default-latest')
Set('StyleSheet/axis-function/autoRange', u'next-tick')
Set('StyleSheet/graph/leftMargin', u'1.5cm')
Set('StyleSheet/graph/rightMargin', u'0.5cm')
Set('StyleSheet/graph/topMargin', u'0.5cm')
Set('StyleSheet/graph/bottomMargin', u'1.5cm')
Set('StyleSheet/page/width', u'16cm')
Set('StyleSheet/page/height', u'10cm')
Set('StyleSheet/xy/marker', u'none')"""


class Dataset:
    def __init__(self, fpath, name=None):
        """A container for the attributes of a dataset"""
        self.path = fpath
        if name:
            self.name = name
        else:
            self.name = fpath.stem

    def x(self):
        return f"{self.name}_col1"

    def y(self):
        return f"{self.name}_col2"


class Page:
    def __init__(self, name=None, x_lower=None, x_upper=None, x_label=None, y_label=None, key=False):
        """A container for the attributes of a Veusz page."""
        if name:
            self.name = name
        else:
            self.name = "plot"
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.x_label = x_label
        self.y_label = y_label
        self.key = key

    def render(self) -> str:
        """Creates a new page with axes ready for graphs."""
        lines = "\n".join([
            f"Add('page', name=u'{self.name}', autoadd=False)",
            f"To(u'{self.name}')",
            "Add('graph', name='graph1', autoadd=False)",
            "To(u'graph1')",
            "Add('axis', name=u'x', autoadd=False)",
            "Add('axis', name=u'y', autoadd=False)",
            "To(u'y')",
            "Set('direction', u'vertical')",
            "To('..')"])
        optional_lines = []
        if self.key:
            tmp = [
                "Add('key', name=u'key1', autoadd=False)",
                "To(u'key1')",
                "Set('Text/size', u'10pt')",
                "Set('Border/hide', True)",
                "To('..')"
            ]
            optional_lines.extend(tmp)
        if self.x_lower is not None:
            tmp = [
                "To(u'x')",
                f"Set('min', {self.x_lower})",
                "To('..')"
            ]
            optional_lines.extend(tmp)
        if self.x_upper is not None:
            tmp = [
                "To(u'x')",
                f"Set('max', {self.x_upper})",
                "To('..')"
            ]
            optional_lines.extend(tmp)
        if self.x_label is not None:
            tmp = [
                "To(u'x')",
                f"Set('label', u'{self.x_label}')",
                "To('..')"
            ]
            optional_lines.extend(tmp)
        if self.y_label is not None:
            tmp = [
                "To(u'y')",
                f"Set('label', u'{self.y_label}')",
                "To('..')"
            ]
            optional_lines.extend(tmp)
        return "\n".join([lines, "\n".join(optional_lines)])


class Graph:
    def __init__(self, name, x_ds, y_ds, color=None, thickness=None):
        """A single graph on a Veusz page."""
        self.name = name
        self.x_name = x_ds
        self.y_name = y_ds
        self.color = color
        self.thickness = thickness

    def render(self) -> str:
        """Render the graph to a string of commands."""
        lines = [
            f"Add('xy', name=u'{self.name}', autoadd=False)",
            f"To(u'{self.name}')",
            f"Set('xData', u'{self.x_name}')",
            f"Set('yData', u'{self.y_name}')",
            f"Set('key', u'{self.name}')"
        ]
        if self.color:
            lines.append(f"Set('color', u'{self.color}')")
        if self.thickness:
            lines.append(f"Set('PlotLine/width', u'{self.thickness:.2f}pt')")
        lines.append("To('..')")
        return "\n".join(lines)


def load_csv(ds) -> str:
    """The statement that imports a CSV file into Veusz.
    """
    return f"ImportFileCSV(u'{ds.path.resolve()}', linked=True, dsprefix=u'{ds.name}_')"


def load_npy(ds) -> str:
    """The statement that imports a NumPy NPY file into Veusz.
    """
    return f"ImportFilePlugin(u'Numpy NPY import', u'{ds.path.resolve()}', linked=True, name=u'{ds.name}')"


def plot_separate(output_file, files, opts):
    """Create a Veusz file where each input file is a separate plot."""
    chunks = [style_settings]
    datasets = [Dataset(f) for f in files]
    chunks.extend([load_csv(d) for d in datasets])
    for d in datasets:
        plot_opts = opts
        plot_opts["name"] = d.name
        page_lines = Page(**plot_opts).render()
        graph_lines = Graph("plot", d.x(), d.y()).render()
        chunks.append("\n".join([page_lines, graph_lines, "To('..')", "To('..')"]))
    contents = "\n".join(chunks)
    with output_file.open("w") as f:
        f.write(contents)


def plot_combined(output_file, files, opts):
    """Create a Veusz file where all the data is on a single plot."""
    chunks = [style_settings]
    datasets = [Dataset(f) for f in files]
    chunks.extend([load_csv(d) for d in datasets])
    graphs = [Graph(d.name, d.x(), d.y()) for d in datasets]
    chunks.append(page_with_graphs(graphs, opts))
    contents = "\n".join(chunks)
    with output_file.open("w") as f:
        f.write(contents)


def page_with_graphs(graphs, opts) -> str:
    """A page made from pre-assembled graphs."""
    chunks = [Page(**opts).render()]
    chunks.extend([g.render() for g in graphs])
    chunks.extend(["To('..')", "To('..')"])
    return "\n".join(chunks)


def plot_gfit(raw_files, fit_files, spectra_files, output_file):
    """Plot comparison graphs for a global fit."""
    raw_datasets = [Dataset(f, f"{f.stem}_raw") for f in raw_files]
    fit_datasets = [Dataset(f, f"{f.stem}_fit") for f in fit_files]
    spectra_datasets = [Dataset(f) for f in spectra_files]
    chunks = [style_settings]
    chunks.extend([load_csv(d) for d in chain(raw_datasets, fit_datasets, spectra_datasets)])
    spectra_page_opts = {
        "x_label": "Wavelength (nm)",
        "x_lower": 780,
        "x_upper": 850,
        "name": "spectra",
        "key": True,
    }
    spectra_graphs = [Graph(d.name, d.x(), d.y(), thickness=1) for d in spectra_datasets]
    chunks.append(page_with_graphs(spectra_graphs, spectra_page_opts))
    curve_opts = {
        "x_label": "Time (us)",
        "x_lower": -10,
        "x_upper": 100
    }
    for raw, fit in zip(raw_datasets, fit_datasets):
        page_name = raw.name[:5]
        curve_opts["name"] = page_name
        raw_graph = Graph("raw", raw.x(), raw.y(), color="black")
        fit_graph = Graph("fit", fit.x(), fit.y(), color="red", thickness=1)
        chunks.append(page_with_graphs([fit_graph, raw_graph], curve_opts))
    contents = "\n".join(chunks)
    with output_file.open("w") as file:
        file.write(contents)


def dataset_groups_from_dirs(dirs, labels=None):
    """Group corresponding datasets from a collection of directories."""
    if labels:
        datasets = [
            [Dataset(f, name=f"{f.stem}_{label}") for f in sorted(d.iterdir()) if f.suffix == ".txt"]
            for d, label in zip(dirs, labels)
        ]
    else:
        datasets = [
            [Dataset(f) for f in sorted(d.iterdir()) if f.suffix == ".txt"]
            for d in dirs
        ]
    # Collects the first datasets from each list into a new list, then the second
    # into a new list, etc.
    groups = [list(x) for x in zip(*datasets)]
    return groups


def plot_compared(dirs, output_file, labels, options):
    """Plot directories of files compared against one another."""
    ds_groups = dataset_groups_from_dirs(dirs, labels=labels)
    chunks = [style_settings]
    chunks.extend([
        load_csv(d) for d in [g for g in chain(*ds_groups)]
    ])
    for i, g in enumerate(ds_groups):
        graphs = [Graph(d.name, d.x(), d.y()) for d in g]
        page_opts = options
        page_opts["name"] = f"plot_{i}"
        page_opts["key"] = True
        chunks.append(page_with_graphs(graphs, options))
    contents = "\n".join(chunks)
    with output_file.open("w") as file:
        file.write(contents)
