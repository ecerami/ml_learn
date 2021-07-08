"""Run the Twitter Disaster Classification Pipeline."""
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
import progressbar


class TwitterPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        train_file = "data/twitter/train.csv"
        test_file = "data/twitter/test.csv"
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            df = self.read_file(train_file)
            train_y = df["target"]
            text_list = df["text"]
            token_list = self.transform_text(text_list)
            print("Creating Corpus.")
            vectorizer = CountVectorizer()
            vectorizer.fit(token_list)

            train_X = vectorizer.transform(token_list)
            self.assess_ml_options(train_y, train_X)
        else:
            print("Train/test files are missing.  Download data first.")

    def assess_ml_options(self, train_y, train_X):
        print("Assessing ML Options.")
        knn = KNeighborsClassifier()
        self.assess_model("KNeighbors", knn, train_X, train_y)

        sgd = SGDClassifier()
        self.assess_model("SGD", sgd, train_X, train_y)

        svc = SVC()
        self.assess_model("Support Vector", svc, train_X, train_y)

        ada = AdaBoostClassifier()
        self.assess_model("Ada Boost", ada, train_X, train_y)

        gbc = GradientBoostingClassifier()
        self.assess_model("Gradient Boost", gbc, train_X, train_y)

        rfc = RandomForestClassifier()
        self.assess_model("Random Forest", rfc, train_X, train_y)

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

    def assess_model(self, name, model, train_X, train_y):
        train_y_pred = cross_val_predict(
            model,
            train_X,
            train_y,
            cv=3,
            verbose=0,
        )
        accuracy = accuracy_score(train_y, train_y_pred)
        precision = precision_score(train_y, train_y_pred)
        recall = recall_score(train_y, train_y_pred)
        f1 = f1_score(train_y, train_y_pred)
        auc = roc_auc_score(train_y, train_y_pred)
        print(name)
        print(f" - Accuracy:  {accuracy:.4f}")
        print(f" - Precision:  {precision:.4f}")
        print(f" - Recall:  {recall:.4f}")
        print(f" - F1:  {f1:.4f}")
        print(f" - AUC:  {auc:.4f}")
        print(confusion_matrix(train_y, train_y_pred))
