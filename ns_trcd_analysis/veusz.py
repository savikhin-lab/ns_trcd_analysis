style_settings = """Set('colorTheme', u'default-latest')
Set('StyleSheet/axis-function/autoRange', u'next-tick')
Set('StyleSheet/graph/leftMargin', u'1.5cm')
Set('StyleSheet/graph/rightMargin', u'0.5cm')
Set('StyleSheet/graph/topMargin', u'0.5cm')
Set('StyleSheet/graph/bottomMargin', u'1.5cm')
Set('StyleSheet/page/width', u'16cm')
Set('StyleSheet/page/height', u'10cm')
Set('StyleSheet/xy/marker', u'none')"""


class Page:
    def __init__(self, name, x_lower=None, x_upper=None, x_label=None, y_label=None, key=False):
        """A container for the attributes of a Veusz page."""
        self.name = name
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
    def __init__(self, name, x_ds, y_ds):
        """A single graph on a Veusz page."""
        self.name = name
        self.x_name = x_ds
        self.y_name = y_ds

    def render(self) -> str:
        """Render the graph to a string of commands."""
        lines = "\n".join([
            f"Add('xy', name=u'{self.name}', autoadd=False)",
            f"To(u'{self.name}')",
            f"Set('xData', u'{self.x_name}')",
            f"Set('yData', u'{self.y_name}')",
            f"Set('key', u'{self.name}')",
            "To('..')"
        ])
        return lines


def load_csv(fname) -> str:
    """The statement that imports a CSV file into Veusz.
    """
    return f"ImportFileCSV(u'{fname.resolve()}', linked=True, dsprefix=u'{fname.stem}_')"


def load_npy(fname) -> str:
    """The statement that imports a NumPy NPY file into Veusz.
    """
    return f"ImportFilePlugin(u'Numpy NPY import', u'{fname.resolve()}', linked=True, name=u'{fname.stem}')"


def plot_separate(output_file, files, opts):
    """Create a Veusz file where each input file is a separate plot."""
    chunks = [style_settings]
    for f in files:
        chunks.append(load_csv(f))
    for f in files:
        page_lines = Page(f.stem, **opts).render()
        graph_lines = Graph("plot", f"{f.stem}_col1", f"{f.stem}_col2").render()
        chunks.append("\n".join([page_lines, graph_lines, "To('..')", "To('..')"]))
    contents = "\n".join(chunks)
    with output_file.open("w") as f:
        f.write(contents)


def plot_combined(output_file, files, opts):
    """Create a Veusz file where all the data is on a single plot."""
    chunks = [style_settings]
    for f in files:
        chunks.append(load_csv(f))
    chunks.append(Page("plot", **opts).render())
    for f in files:
        graph = Graph(f"{f.stem}", f"{f.stem}_col1", f"{f.stem}_col2")
        chunks.append(graph.render())
    chunks.append("To('..')")
    chunks.append("To('..')")
    contents = "\n".join(chunks)
    with output_file.open("w") as f:
        f.write(contents)
