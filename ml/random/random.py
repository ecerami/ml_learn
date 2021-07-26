"""
Run the Random Number Generator Pipeline.

The pipeline illustrates the basics of how to generate
random numbers from different distributions.

"""
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt


class RandomNumberGeneratorPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        # default_rng takes an optional seed value
        rng = default_rng(seed=42)

        # for different distributions
        # see: https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng

        vals = rng.uniform(0, 10, 1000)
        self.random_distribution("Uniform", vals)

        vals = rng.standard_normal(1000)
        self.random_distribution("Standard Normal", vals)

        # Let's try a fake linear regresssion data set
        print("Linear Regression Example")
        # x = random numbers between 0 and 2
        x = 2 * rng.uniform(0, 1, 100)
        # y-intercept = 4
        # slope = 3
        # and add a bit of normal noise
        y = 4 + 3 * x + rng.standard_normal(100)
        df = pd.DataFrame()
        df["x"] = x
        df["y"] = y
        df.plot.scatter(x="x", y="y")
        plt.show()

    def random_distribution(self, name, vals):
        df = pd.DataFrame()
        df["vals"] = vals
        print(name)
        df.hist(bins=50)
        plt.show()
