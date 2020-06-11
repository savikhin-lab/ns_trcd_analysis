import click
from pathlib import Path
from . import raw2hdf5


@click.group()
def cli():
    pass


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("outfile_name", type=click.Path(exists=False, file_okay=True, dir_okay=False))
def assemble(input_dir, outfile_name):
    in_dir = Path(input_dir)
    outfile = in_dir / outfile_name
    raw2hdf5.ingest(in_dir, outfile)


@click.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_file", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("-a", "--average", is_flag=True, help="Average dA and save the result")
def da(input_file, output_file, average):
    print("da")


cli.add_command(assemble)
cli.add_command(da)
