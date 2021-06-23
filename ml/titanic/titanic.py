"""Run the Titanic Classification Pipeline."""
from ml.titanic.titanic_prep import TitanicPrep
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class TitanicPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        print("Loading Titanic Training Data Set.")
        train_df = pd.read_csv("data/titanic_train.csv")
        print(f"Loaded data frame: {train_df.shape[0]} x {train_df.shape[1]}.")
        train_y = train_df["Survived"]
        train_X = train_df.drop("Survived", axis=1)

        print("Transforming training set.")
        prep = TitanicPrep()
        prep.fit(train_X)
        train_X = prep.transform(train_X)

        knn = KNeighborsClassifier()
        self.assess_model("KNeighbors", knn, train_X, train_y)

        sgd = SGDClassifier()
        self.assess_model("SGD", sgd, train_X, train_y)

        gnb = GaussianNB()
        self.assess_model("Naive Bayes", gnb, train_X, train_y)

        svc = SVC()
        self.assess_model("Support Vector", svc, train_X, train_y)

        ada = AdaBoostClassifier()
        self.assess_model("Ada Boost", ada, train_X, train_y)

        rfc = RandomForestClassifier()
        self.assess_model("Random Forest", rfc, train_X, train_y)

        print("Predicting on test data set.")
        test_df = pd.read_csv("data/titanic_test.csv")
        passengerIds = test_df["PassengerId"]
        test_X = prep.transform(test_df)
        svc.fit(train_X, train_y)
        survived = svc.predict(test_X)
        submission_df = pd.DataFrame()
        submission_df["PassengerId"] = passengerIds
        submission_df["Surived"] = survived
        out_name = "out/titanic_predictions.csv"
        print (f"Writing predicions to {out_name}.")
        submission_df.to_csv(out_name, sep=",", index=False)

    def assess_model(self, name, model, train_X, train_y):
        results = cross_val_score(
            model,
            train_X,
            train_y,
            cv=3,
            verbose=0,
            scoring="accuracy",
        )
        print(f"{name} = {results.mean():.4f}")
