"""Run the Twitter Pre-Processing Pipeline."""
from ml.twitter.twitter_parser import TwitterParser
import os.path
import pandas as pd
import progressbar


class TwitterPreparePipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        train_file = "data/twitter/train.csv"
        test_file = "data/twitter/test.csv"
        train_out_file = "out/twitter_train.csv"
        test_out_file = "out/twitter_test.csv"
        self.transform_file(train_file, train_out_file)
        self.transform_file(test_file, test_out_file)            

    def transform_file(self, in_file, out_file):
        if os.path.isfile(in_file):
            df = self.read_file(in_file)
            text_list = df["text"]
            token_list = self.transform_text(in_file, out_file, text_list)
            df["text_prepared"] = token_list
            df.to_csv(out_file, sep=",", index=False)
        else:
            print(f"File: {in_file} is missing.  Download data first.")

    def transform_text(self, in_file, out_file, text_list):
        twitter_parser = TwitterParser()
        print(f"Pre-processing Twitter messages:  {in_file} --> {out_file}")
        token_list = []
        for i in progressbar.progressbar(range(len(text_list))):
            text = text_list[i]
            token_str = " ".join(twitter_parser.normalize(text))
            token_list.append(token_str)
        return token_list

    def read_file(self, file_name):
        df = pd.read_csv(file_name)
        return df
