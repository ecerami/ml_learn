"""
California Housing Price Regression.

Our goal is to take as input multiple parameters, including:
- latitude and longitude
- median age
- total rooms
- total bedrooms
- population
- households
- median income
- ocean proximity

and predict:  median house value.
"""
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


class HousingRegressionPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        # ML Workflow
        df = self.load_data()
        train_set, test_set = self.split_data(df)

        # Prepare Training Set
        LABEL_NAME = "median_house_value"
        training_labels = train_set[LABEL_NAME]
        training_set = train_set.drop(LABEL_NAME, axis=1)
        prepared_training_df = self.prepare_data(training_set)

        # Assess Different Model Options
        self.assess_model_options(prepared_training_df, training_labels)

    def load_data(self):
        return pd.read_csv("data/housing.csv")

    def split_data(self, df):
        return train_test_split(df, test_size=0.2, random_state=42)

    def prepare_data(self, df):
        housing_num = df.drop("ocean_proximity", axis=1)
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]

        # Replace missing values with median.
        # Replace all values with standard scaling.
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("std_scaler", StandardScaler()),
            ]
        )

        # Replace Ocean Proximity with One Hot Encoding
        full_pipeline = ColumnTransformer(
            [
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ]
        )

        data_prepared = full_pipeline.fit_transform(df)
        return data_prepared


    # Create Model Options
    def assess_model_options(self, training_df, train_labels):
        model1 = LinearRegression()
        model2 = RandomForestRegressor()
        model3 = svm.SVR(kernel="linear")
        model4 = svm.SVR(kernel = "rbf")

        self.assess_model("Linear", model1, training_df, train_labels)
        self.assess_model("Random Forest", model2, training_df, train_labels)
        self.assess_model("SVR: Linear", model3, training_df, train_labels)
        self.assess_model("SVR: RBF", model4, training_df, train_labels)


    def assess_model(self, model_name, model, training_df, train_labels):
        start = time.time()
        model.fit(training_df, train_labels)
        scores = cross_val_score(
            model,
            training_df,
            train_labels,
            scoring="neg_mean_squared_error",
            cv=4,
        )
        mean_score = np.sqrt(-scores).mean()
        stop = time.time()
        duration = stop - start
        print(f"{model_name} -> {mean_score:0.2f} [{duration:0.2f} sec]")
