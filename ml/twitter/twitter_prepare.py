"""Run the Twitter Pre-Processing Pipeline."""
from ml.twitter.twitter_parser import TwitterParser
import os.path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import progressbar


class TwitterPreparePipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        train_file = "data/twitter/train.csv"
        test_file = "data/twitter/test.csv"
        train_out_file = "out/twitter_train.csv"
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            df = self.read_file(train_file)
            text_list = df["text"]
            token_list = self.transform_text(text_list)
            df["text_prepared"] = token_list
            df.to_csv(train_out_file, sep=",", index=False)
            print(f"Writing to:  {train_out_file}.")
        else:
            print("Train/test files are missing.  Download data first.")

    def transform_text(self, text_list):
        print("Pre-processing Twitter messages.")
        token_list = []
        for i in progressbar.progressbar(range(len(text_list))):
            text = text_list[i]
            twitter_parser = TwitterParser(text)
            token_str = " ".join(twitter_parser.get_final_token_list())
            token_list.append(token_str)
        return token_list

    def read_file(self, file_name):
        df = pd.read_csv(file_name)
        return df

