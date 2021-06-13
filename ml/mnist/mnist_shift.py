"""Run the Mnist Pipeline with an Augmented Training Set of Shifted Images."""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage.interpolation import shift
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import progressbar


class MnistShiftPipeline:
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

        X_augmented_train, y_augmented_train = self.augment_training_set(
            X_train, y_train
        )

        print(f"Augmented feature set: {X_augmented_train.shape[0]} ", end="")
        print(f"x {X_augmented_train.shape[1]}.")
        print(f"Augmented labels: {len(y_augmented_train)}.")

        print("Cross-Validating KNeighbors Classifier.")
        knn = KNeighborsClassifier(n_neighbors=4, weights="distance")
        results = cross_val_score(
            knn,
            X_augmented_train,
            y_augmented_train,
            cv=3,
            verbose=3,
            scoring="accuracy",
        )

        print("Cross-validation accuracy scores:")
        for result in results:
            print(result)

        print("Rebuilding KNN Model")
        knn.fit(X_augmented_train, y_augmented_train)

        print("Assessing against test set.")
        y_pred = knn.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test data set:  {test_accuracy}")

    def augment_training_set(self, X_train, y_train):
        """Augment training set with shifted images."""
        X_augmented_train = []
        y_augmented_train = []
        print("Augmenting data set with shifted images.")
        for i in progressbar.progressbar(range(len(X_train))):
            current_x = X_train.iloc[[i]].to_numpy()
            current_y = y_train.iloc[[i]].to_numpy()[0]

            # Get Current Digit
            current_digit = current_x.reshape(28, 28)

            # Shift Digit in each direction
            shift1 = shift(current_digit, [0, 1], cval=0)
            shift2 = shift(current_digit, [0, -1], cval=0)
            shift3 = shift(current_digit, [1, 0], cval=0)
            shift4 = shift(current_digit, [-1, 0], cval=0)

            # Augment to X and y
            X_augmented_train.append(current_digit.reshape(784))
            X_augmented_train.append(shift1.reshape(784))
            X_augmented_train.append(shift2.reshape(784))
            X_augmented_train.append(shift3.reshape(784))
            X_augmented_train.append(shift4.reshape(784))
            for _ in range(5):
                y_augmented_train.append(current_y)

        print("Creating new data frames.  This may take a few minutes...")
        X_df = pd.DataFrame(X_augmented_train)
        y_df = pd.Series(y_augmented_train)
        return (X_df, y_df)
