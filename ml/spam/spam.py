"""Run the Spam Classification Pipeline."""
from sklearn.feature_extraction.text import CountVectorizer
import os.path


class SpamPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        train_file = "out/spam_train.txt"
        test_file = "out/spam_test.txt"
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            spam_list, token_list = self.read_file(train_file)
            print("Creating Corpus.")
            vectorizer = CountVectorizer()
            vectorizer.fit(token_list)

            self.write_corpus(vectorizer)
            X = vectorizer.transform(token_list)

        else:
            print("Train/test files are missing.  First run:  ml spam-prepare")

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
            spam_list.append(parts[0].strip())
            token_list.append(parts[2].strip())
        return (spam_list, token_list)
