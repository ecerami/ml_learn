# Titanic Notes

## Experiment #1

Tried six ML Models:  KNeighborsClassifier, SGDClassifier, GaussianNB,
Support Vector (SVC), AdaBoostClassifier, RandomForestClassifier.

Results:

KNeighbors = 0.7946
SGD = 0.7755
Naive Bayes = 0.7666
Support Vector = 0.8114
Ada Boost = 0.7969
Random Forest = 0.7991

Went with SVC for Kaggle, and got 0.77751.

## Experiment #2

First round of feature engineering.  Binarize age into child v. adult.

KNeighbors = 0.8070
SGD = 0.6723
Naive Bayes = 0.7699
Support Vector = 0.8171
Ada Boost = 0.7789
Random Forest = 0.7980

SVC is still the winner, and it is only marginally better.

Kaggle score:  0.77511 (no change!)

Verdict:  didn't help any.

But, changing the threshold to 12.0 seemed to help a tiny bit more:

Support Vector = 0.8260

## Experiment #3

Switched from MinMaxScaler to StandardScaler.

KNeighbors = 0.8103
SGD = 0.7912
Naive Bayes = 0.7755
Support Vector = 0.8305
Ada Boost = 0.7856
Random Forest = 0.8070

SVC still the winner, but is only marginally better.

Kaggle score:  0.77751 (no change again!)

# Experiment #4

Used GridSearch to try to optimize three models:

Executing GridSearch to determine best KNN parameters.
Best Parameter:  n_neighbors --> 4
Best Parameter:  weights --> uniform
Best score:  0.8092461239093591
Executing GridSearch to determine best RFC parameters.
Best Parameter:  criterion --> gini
Best Parameter:  n_estimators --> 50
Best score:  0.8159751428033394
Executing GridSearch to determine best SVC parameters.
Best Parameter:  C --> 0.5
Best Parameter:  kernel --> poly
Best score:  0.8282781997363632

SVC is still the top scoring option.

Kaggle score:  0.77751 (no change again!)

# Experment #5

Changed PClass back to a numerical value, because it is actually ordinal.

Executing GridSearch to determine best SVC parameters.
Best Parameter:  C --> 1.5
Best Parameter:  kernel --> poly
Best score:  0.8305128366078716

SVC is marginally up.

Kaggle score:  0.77990 (finally, an increase!)

# Experiment #7

Reverted age back to continuous variable instead of binarized.

Executing GridSearch to determine best SVC parameters.
Best Parameter:  C --> 2
Best Parameter:  kernel --> rbf
Best score:  0.831648986253217

SVC is still best, but now marginally up.

Kaggle score:  0.78229 (an increase!)

# Experiment #8

Extracted deck level from Cabin column, e.g. B222 becomes Deck_B.

Executing GridSearch to determine best SVC parameters.
Best Parameter:  C --> 1.5
Best Parameter:  kernel --> rbf
Best score:  0.8260372857949909

SCV is still best, but now marginally down.

Kaggle score:  0.77511 (also down!)

This is probably not too surprising given that most 2nd and 3rd class
passengers have missing Cabin values; so, this may just really be a proxy for
the already existing Pclass column.

# Experiment #9

Found the best age cutoff to be 13.

Executing GridSearch to determine best SVC parameters.
  - Best Parameter:  C --> 2
  - Best Parameter:  kernel --> rbf
  - Best score:  0.8327600276191074

Kaggle scoere:  0.77511 (but, this does not beat 0.78229 for age as continous.)
So, not sure this was worth it.
