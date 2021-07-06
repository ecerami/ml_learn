"""
ML Command Line Interface (CLI).
"""
import logging
from ml.spam.spam_prepare import SpamPreparePipeline
from ml.spam.spam import SpamPipeline
from ml.titanic.titanic import TitanicPipeline
from ml.mnist.mnist_shift import MnistShiftPipeline
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


@cli.command()
def mnist_augment():
    """Run the MNIST Classifier Pipeline with Augmented Data."""
    output_header("Running the MNIST Classifier Pipeline with Augmented Data.")
    pipeline = MnistShiftPipeline()
    pipeline.execute_pipeline()


@cli.command()
def titanic():
    """Run the Titanic Classifier Pipeline."""
    output_header("Running the Titanic Classifier Pipeline.")
    pipeline = TitanicPipeline()
    pipeline.execute_pipeline()


@cli.command()
def spam_prepare():
    """Run the Spam Pre-Processor Pipeline."""
    output_header("Running the Spam Pre-Processor Pipeline.")
    pipeline = SpamPreparePipeline()
    pipeline.execute_pipeline()


@cli.command()
def spam():
    """Run the Spam Classification Pipeline."""
    output_header("Running the Spam Classification Pipeline.")
    pipeline = SpamPipeline()
    pipeline.execute_pipeline()


def output_header(msg):
    """Output header with emphasis."""
    click.echo(click.style(msg, fg="green"))


def output_error(msg):
    """Output error message with emphasis."""
    msg = emoji.emojize(f":warning:  {msg}", use_aliases=True)
    click.echo(click.style(msg, fg="yellow"))
