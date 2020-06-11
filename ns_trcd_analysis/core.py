def count_subdirs(path) -> int:
    """Count the number of subdirectories directly under `path`.

    This is useful for determining the number of wavelengths and shots in an experiment.
    """
    count = 0
    for item in path.iterdir():
        if item.is_dir():
            count += 1
    return count
