"""Prepare the Titanic Data Set."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class TitanicPrep:
    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
    num_columns = ["SibSp", "Parch", "Fare"]
    cat_columns = ["Pclass", "Sex"]

    def __init__(self):
        """Construct prep pipeline."""

    def fit(self, training_df):
        age_pipeline = Pipeline(
            [
                ("impute_age", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler()),
            ]
        )
        embark_pipeline = Pipeline(
            [
                ("impute_embark", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder()),
            ]
        )
        num_pipeline = Pipeline([("scale", MinMaxScaler())])
        cat_pipeline = Pipeline([("ohe", OneHotEncoder())])

        self.pipeline = ColumnTransformer(
            [
                ("drop", "drop", self.drop_columns),
                ("embark", embark_pipeline, ["Embarked"]),
                ("cat", cat_pipeline, self.cat_columns),
                ("num", num_pipeline, self.num_columns),
                ("age", age_pipeline, ["Age"]),
            ]
        )
        self.pipeline.fit(training_df)

    def transform(self, X):
        new_X = pd.DataFrame(self.pipeline.transform(X))

        # Hack to get all the new column names
        emb_list = self.pipeline.transformers_[1][1][1].get_feature_names()
        cat_list = self.pipeline.transformers_[2][1][0].get_feature_names()
        num_list = self.num_columns
        emb_list = [x.replace("x0", "Embarked") for x in emb_list]
        cat_list = [x.replace("x0", "Class") for x in cat_list]
        cat_list = [x.replace("x1", "Sex") for x in cat_list]

        new_columns = []
        new_columns.extend(emb_list)
        new_columns.extend(cat_list)
        new_columns.extend(num_list)
        new_columns.append("Age")
        new_X.columns = new_columns
        return new_X
