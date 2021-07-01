import numpy as np
import pandas as pd
from nameparser import HumanName
from sklearn.impute import SimpleImputer


class AgeImputer(SimpleImputer):
    """
    Age Imputer.
    If age is missing, impute age based on title.
    For example, if the name refers to "Master.", this is likely a child.
    If this is a "Dr." or "Rev.", this is likely not a child.
    """

    def fit(self, X, y=None):
        names = X["Name"]
        title_list = []
        for name in names:
            # Leverage HumanName Parser
            # see:  https://github.com/derek73/python-nameparser
            human_name = HumanName(name)
            title = human_name["title"]
            if len(title) == 0:
                title = "Other"
            title_list.append(title)

        sub_df = pd.DataFrame()
        sub_df["Title"] = title_list
        sub_df["Age"] = X["Age"]
        self.median_by_title = sub_df.groupby(by="Title")["Age"].median()
        return self

    def transform(self, X):
        name_list = X["Name"]
        age_list = X["Age"]
        inferred_age_list = []
        for i in range(0, len(name_list)):
            name = name_list[i]
            age = age_list[i]
            if np.isnan(age):
                human_name = HumanName(name)
                title = human_name["title"]
                if len(title) == 0:
                    title = "Other"
                inferred_age = self.median_by_title[title]
                inferred_age_list.append(inferred_age)
            else:
                inferred_age_list.append(age)
        X["Age"] = inferred_age_list
        return X
