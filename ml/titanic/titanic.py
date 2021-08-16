"""Run the Titanic Classification Pipeline."""
from ml.titanic.titanic_prep import TitanicPrep
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class TitanicPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        print("Loading Titanic Training Data Set.")
        train_df = pd.read_csv("data/titanic/train.csv")
        print(f"Loaded data frame: {train_df.shape[0]} x {train_df.shape[1]}.")
        train_y = train_df["Survived"]
        train_X = train_df.drop("Survived", axis=1)

        print("Transforming training set.")
        prep = TitanicPrep()
        prep.fit(train_X)
        train_X = prep.transform(train_X)

        print("Assessing ML Options.")
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

        gbc = GradientBoostingClassifier()
        self.assess_model("Gradient Boost", gbc, train_X, train_y)

        rfc = RandomForestClassifier()
        self.assess_model("Random Forest", rfc, train_X, train_y)

        xgb = XGBClassifier(use_label_encoder=False, eval_metric="error")
        self.assess_model("XGBoost", xgb, train_X, train_y)

        logit = LogisticRegression()
        self.assess_model("Logistic Regression", logit, train_X, train_y)

        print("Executing GridSearch to determine best KNN parameters.")
        weight_list = ["uniform", "distance"]
        neighbor_list = [3, 4, 5]
        param_grid = [{"weights": weight_list, "n_neighbors": neighbor_list}]
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=0)
        grid_search.fit(train_X, train_y)
        self._output_grid_search_results(grid_search)

        print("Executing GridSearch to determine best RFC parameters.")
        n_estimators = [50, 100, 150]
        criterion_list = ["gini", "entropy"]
        param_grid = [
            {"n_estimators": n_estimators, "criterion": criterion_list}
        ]
        rfc = RandomForestClassifier()
        grid_search = GridSearchCV(rfc, param_grid, cv=5, verbose=0)
        grid_search.fit(train_X, train_y)
        self._output_grid_search_results(grid_search)

        print("Executing GridSearch to determine best SVC parameters.")
        c_list = [0.25, 0.5, 1.0, 1.5, 2]
        kernel_list = ["linear", "poly", "rbf", "sigmoid"]
        param_grid = [{"C": c_list, "kernel": kernel_list}]
        svc = SVC()
        grid_search = GridSearchCV(svc, param_grid, cv=5, verbose=0)
        grid_search.fit(train_X, train_y)
        self._output_grid_search_results(grid_search)

        self._determine_most_predictive_features(train_y, train_X)

        print("Predicting on test data set.")
        test_df = pd.read_csv("data/titanic/test.csv")
        passengerIds = test_df["PassengerId"]
        test_X = prep.transform(test_df)
        svc = SVC(C=1.5, kernel="rbf")
        svc.fit(train_X, train_y)
        survived = svc.predict(test_X)
        submission_df = pd.DataFrame()
        submission_df["PassengerId"] = passengerIds
        submission_df["Survived"] = survived
        out_name = "out/titanic_predictions.csv"
        print(f"Writing predictions to {out_name}.")
        submission_df.to_csv(out_name, sep=",", index=False)

    def _output_grid_search_results(self, grid_search):
        best_params = grid_search.best_params_
        for param in best_params:
            print(f"  - Best Parameter:  {param} --> {best_params[param]}")
        print(f"  - Best score:  {grid_search.best_score_}")

    def _determine_most_predictive_features(self, train_y, train_X):
        print("Determining most predictive features.")
        svc = SVC(kernel="linear")
        svc.fit(train_X, train_y)
        coef = svc.coef_[0]
        feature_names = train_X.columns
        predictors = list(zip(feature_names, coef))
        predictors.sort(key=lambda x: abs(x[1]), reverse=True)
        for predictor in predictors:
            print(f"  - {predictor[0]}:  {predictor[1]}")

    def assess_model(self, name, model, train_X, train_y):
        results = cross_val_score(
            model,
            train_X,
            train_y,
            cv=3,
            verbose=0,
            scoring="accuracy",
        )
        print(f"  - {name} = {results.mean():.4f}")
