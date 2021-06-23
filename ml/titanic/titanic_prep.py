"""Prepare the Titanic Data Set."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class TitanicPrep:
    """Prepare the Titanic Data Set."""

    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
    num_columns = ["SibSp", "Parch", "Age", "Fare"]
    cat_columns = ["Embarked", "Pclass", "Sex"]

    def __init__(self):
        """Construct prep pipeline."""

    def fit(self, training_df):
        impute_scale = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler()),
            ]
        )
        impute_ohe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder()),
            ]
        )

        self.pipeline = ColumnTransformer(
            [
                ("drop", "drop", self.drop_columns),
                ("impute_ohe", impute_ohe, self.cat_columns),
                ("impute_scale", impute_scale, self.num_columns),
            ]
        )
        self.pipeline.fit(training_df)

    def transform(self, X):
        new_X = pd.DataFrame(self.pipeline.transform(X))

        # Hack to get all the new OHE column names
        cat_list = self.pipeline.transformers_[1][1][1].get_feature_names()
        cat_list = [x.replace("x0", "Embarked") for x in cat_list]
        cat_list = [x.replace("x1", "Class") for x in cat_list]
        cat_list = [x.replace("x2", "Sex") for x in cat_list]

        new_columns = []
        new_columns.extend(cat_list)
        new_columns.extend(self.num_columns)
        new_X.columns = new_columns
        return new_X