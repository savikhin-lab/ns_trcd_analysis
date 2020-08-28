def global_fit_file(task_names, lifetimes, amplitudes, first_spec_in, first_spec_out, instr_spec):
    """Generate a global-fit input file.
    """
    sections = []
    sections.append(header(len(task_names), len(lifetimes), lifetimes))
    for i in range(len(task_names)):
        tname = task_names[i]
        spec_in = first_spec_in + i
        spec_out = first_spec_out + i
        sections.append(task(tname, i, spec_in, spec_out, instr_spec, lifetimes, amplitudes))
    contents = "\r\n".join(sections)
    contents += "\r\n"
    return contents


def header(ntasks, nlifetimes, lifetimes) -> str:
    """The global-fit input file header
    """
    lines = []
    lines.append("Global fit input file")
    lines.append(f"Number of tasks = {ntasks}")
    lines.append(f"Global variables = {nlifetimes}")
    for i in range(len(lifetimes)):
        lines.append(f"t{i}")
        lines.append(f"    Initial value = {lifetimes[i]} Change = {lifetimes[i]/10:.2f}")
    header = "\r\n".join(lines)
    return header


def task(name, task_num, spec_in, spec_out, instr_spec, lifetimes, amplitudes):
    """The text for a single curve to be fit.

    The formatting of this file is bizarre. I have no idea why some of the newlines
    are required, they just are. ¯\_(ツ)_/¯
    """
    lines = []
    lines.append(f"****** Task #{task_num}: {name}")
    lines.append(f"Spec in = {spec_in} from 0 to 1000")
    lines.append(f"Instrument func spec = {instr_spec} from -1 to 1")
    lines.append("Background spec = 0")
    lines.append("Mask spec = 0")
    lines.append(f"Spec out = {spec_out}")
    dummy_lifetimes = 6 - len(lifetimes)
    for i in range(len(lifetimes)):
        lines.append(f"Lifetime{i}:")
        lines.append(f"t{i}")
        lines.append("    Change = 0")
        lines.append(f"    Amplitude = \r\n{amplitudes[i]}")
        lines.append(f"    Change = {amplitudes[i]/10:.2f}")
    if dummy_lifetimes != 0:
        start_idx = 6 - dummy_lifetimes + 1
        for i in range(start_idx, 7):
            lines.append(f"Lifetime{i}:")
            lines.append("    ")
            lines.append("    Change = 0")
            lines.append("    Amplitude = \r\n0")
            lines.append("    Change = 0")
    lines.append("Background level = none\r\n")
    lines.append("    change = 0")
    lines.append("Spike level = none\r\n")
    lines.append("    change = 0")
    lines.append("Shift amount = none\r\n")
    lines.append("    change = 0")
    lines.append("Fill amplitude level = none\r\n")
    lines.append("    change = 0")
    lines.append("Noise model = 1")
    lines.append("Task weight = 1")
    lines.append("Automatic task weight = 0")
    task_contents = "\r\n".join(lines)
    return task_contents
