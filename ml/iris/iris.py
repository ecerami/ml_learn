"""Run the Iris Classification Pipeline."""
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

class IrisPipeline:
    def __init__(self):
        """Construct pipeline."""

    def split_data(self, df):
        return train_test_split(df, test_size=0.2, random_state=42)

    def execute_pipeline(self):
        print("Loading Iris Data Set.")
        df = pd.read_csv("data/iris.csv")
        print(f"Loaded data frame: {df.shape[0]} x {df.shape[1]}.")
        train_df, test_df = self.split_data(df)

        train_y = train_df["class"]
        train_X = train_df.drop("class", axis=1)
        print(f"Training Set: {train_df.shape[0]} x {train_df.shape[1]}.")
        print(f"Test Set: {test_df.shape[0]} x {test_df.shape[1]}.")

        print("Assessing ML Options.")
        svc = SVC()
        self.assess_model("Support Vector", svc, train_X, train_y)

        print("Executing GridSearch to determine best SVC parameters.")
        c_list = [0.25, 0.5, 1.0, 1.5, 2]
        kernel_list = ["linear", "poly", "rbf", "sigmoid"]
        param_grid = [{"C": c_list, "kernel": kernel_list}]
        svc = SVC()
        grid_search = GridSearchCV(svc, param_grid, cv=5, verbose=0)
        grid_search.fit(train_X, train_y)
        self._output_grid_search_results(grid_search)

    def _output_grid_search_results(self, grid_search):
        best_params = grid_search.best_params_
        for param in best_params:
            print(f"  - Best Parameter:  {param} --> {best_params[param]}")
        print(f"  - Best score:  {grid_search.best_score_}")

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
