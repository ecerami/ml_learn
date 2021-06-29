import pandas as pd
from ml.titanic.titanic_prep import TitanicPrep


def test_titanic_prep():
    """Test Titatic Data Prep."""
    df = pd.read_csv("data/titanic_train.csv")
    training_X = df.drop("Survived", axis=1)
    prep = TitanicPrep()
    prep.fit(training_X)
    training_X_prepared = prep.transform(training_X)

    # Verify number of new columns
    assert len(training_X_prepared.columns) == 10

    # Verify that one hot encoding worked on the zeroeth row
    row0 = training_X_prepared.iloc[0]
    assert row0.Embarked_C == 0
    assert row0.Embarked_Q == 0
    assert row0.Embarked_S == 1

    assert row0.Pclass == 0.8273772438659699

    assert row0.Sex_female == 0
    assert row0.Sex_male == 1
