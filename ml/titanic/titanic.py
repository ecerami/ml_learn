"""Run the Titanic Classification Pipeline."""
import pandas as pd

# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline


class TitanicPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        print("Loading Titanic Training Data Set.")
        df = pd.read_csv("data/titanic_train.csv")
        print(f"Loaded data frame: {df.shape[0]} x {df.shape[1]}.")
        # y = df["Survived"]
        X = df.drop("Survived", axis=1)
        print(X.head())
