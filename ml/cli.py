"""
ML Command Line Interface (CLI).
"""
import logging
from ml.mnist.mnist import MnistPipeline
import click
import emoji


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose mode")
def cli(verbose):
    """Run ML Pipelines."""
    log_level = logging.FATAL
    if verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(message)s")


@cli.command()
def mnist():
    """Run the MNIST Classifier Pipeline."""
    output_header("Running the MNIST Classifier Pipeline.")
    pipeline = MnistPipeline()
    pipeline.execute_pipeline()


def output_header(msg):
    """Output header with emphasis."""
    click.echo(click.style(msg, fg="green"))


def output_error(msg):
    """Output error message with emphasis."""
    msg = emoji.emojize(f":warning:  {msg}", use_aliases=True)
    click.echo(click.style(msg, fg="yellow"))
