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

SVC is still the winner, and it is marginally better.

Kaggle score:  0.77511 (no change!)

