"""Run the Mnist Pipeline."""
import pandas as pd


class MnistPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        print("Loading MNIST Data Set.")
        df = pd.read_csv("data/mnist_784.csv")
        print(f"Loaded data frame: {df.shape[0]} x {df.shape[1]}.")
