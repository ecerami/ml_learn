"""Run the Spam Classification Pipeline."""
import os.path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score


class SpamPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        train_file = "out/spam_train.txt"
        test_file = "out/spam_test.txt"
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            train_y, token_list = self.read_file(train_file)
            print("Creating Corpus.")
            #vectorizer = CountVectorizer()
            vectorizer = TfidfVectorizer()
            vectorizer.fit(token_list)

            self.write_corpus(vectorizer)
            train_X = vectorizer.transform(token_list)
            self.assess_ml_options(train_y, train_X)

        else:
            print("Train/test files are missing.  First run:  ml spam-prepare")

    def assess_ml_options(self, train_y, train_X):
        print("Assessing ML Options.")
        knn = KNeighborsClassifier()
        self.assess_model("KNeighbors", knn, train_X, train_y)

        sgd = SGDClassifier()
        self.assess_model("SGD", sgd, train_X, train_y)

        svc = SVC()
        self.assess_model("Support Vector", svc, train_X, train_y)

        ada = AdaBoostClassifier()
        self.assess_model("Ada Boost", ada, train_X, train_y)

        gbc = GradientBoostingClassifier()
        self.assess_model("Gradient Boost", gbc, train_X, train_y)

        rfc = RandomForestClassifier()
        self.assess_model("Random Forest", rfc, train_X, train_y)

    def write_corpus(self, vectorizer):
        corpus = "out/corpus.txt"
        print(f"Writing corpus to: {corpus}.")
        corpus_fd = open(corpus, "w")
        feature_names = vectorizer.get_feature_names()
        for i in range(0, len(feature_names)):
            feature_name = feature_names[i]
            out = f"{i}\t{feature_name}\n"
            corpus_fd.write(out)
        corpus_fd.close()

    def read_file(self, file_name):
        fd = open(file_name)
        spam_list = []
        token_list = []
        for line in fd:
            parts = line.split("\t")
            spam_list.append(int(parts[0].strip()))
            token_list.append(parts[2].strip())
        return (spam_list, token_list)

    def assess_model(self, name, model, train_X, train_y):
        train_y_pred = cross_val_predict(
            model,
            train_X,
            train_y,
            cv=3,
            verbose=0,
        )
        accuracy = accuracy_score(train_y, train_y_pred)
        precision = precision_score(train_y, train_y_pred)
        recall = recall_score(train_y, train_y_pred)
        f1 = f1_score(train_y, train_y_pred)
        print(name)
        print(f" - Accuracy:  {accuracy:.4f}")
        print(f" - Precision:  {precision:.4f}")
        print(f" - Recall:  {recall:.4f}")
        print(f" - F1:  {f1:.4f}")
        print(confusion_matrix(train_y, train_y_pred))
