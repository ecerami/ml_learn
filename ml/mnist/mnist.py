"""Run the Mnist Pipeline."""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class MnistPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        print("Loading MNIST Data Set.")
        df = pd.read_csv("data/mnist_784.csv")
        print(f"Loaded data frame: {df.shape[0]} x {df.shape[1]}.")
        y = df["class"]
        X = df.drop("class", axis=1)

        print("Splitting data into training and test set.")
        X_train = X[:60000]
        y_train = y[:60000]
        X_test = X[60000:]
        y_test = y[60000:]
        print(f"Training data frame: {X_train.shape[0]} x {X_train.shape[1]}.")

        print("Creating and Cross-Validating KNeighbors Classifier.")
        print("Executing GridSearch to determine best parameters.")
        weight_list = ["uniform", "distance"]
        neighbor_list = [3, 4, 5]
        param_grid = [{"weights": weight_list, "n_neighbors": neighbor_list}]
        knn_clf = KNeighborsClassifier()
        grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        for param in best_params:
            print(f"Best Parameter:  {param} --> {best_params[param]}")
        print(f"Best score:  {grid_search.best_score_}")

        y_pred = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test data set:  {test_accuracy}")
