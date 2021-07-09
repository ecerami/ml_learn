"""Run the Twitter Disaster Classification Pipeline."""
from ml.twitter.twitter_parser import TwitterParser
import os.path
import pandas as pd

# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier


class TwitterPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        train_file = "out/twitter_train.csv"
        test_file = "out/twitter_test.csv"
        if os.path.isfile(train_file):
            train_df = pd.read_csv(train_file)
            train_y = train_df["target"]
            train_text_list = train_df["text_prepared"]
            print("Creating Corpus.")
            vectorizer = TfidfVectorizer()
            vectorizer.fit(train_text_list)

            train_X = vectorizer.transform(train_text_list)
            self.assess_ml_options(train_y, train_X)
            self.predict_on_test_set(test_file, train_y, vectorizer, train_X)
        else:
            print("Train/test files are missing.  Run twitter-prepare first.")

    def predict_on_test_set(self, test_file, train_y, vectorizer, train_X):
        test_df = pd.read_csv(test_file)
        test_text_list = test_df["text_prepared"]
        test_X = vectorizer.transform(test_text_list)
        sgd = SGDClassifier()
        sgd.fit(train_X, train_y)
        disaster = sgd.predict(test_X)
        submission_df = pd.DataFrame()
        submission_df["id"] = test_df["id"]
        submission_df["target"] = disaster
        out_name = "out/twitter_predictions.csv"
        print(f"Writing predictions to {out_name}.")
        submission_df.to_csv(out_name, sep=",", index=False)

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

        xgb = XGBClassifier(use_label_encoder=False, eval_metric="error")
        self.assess_model("XGBoost", xgb, train_X, train_y)

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
        #print(confusion_matrix(train_y, train_y_pred))
