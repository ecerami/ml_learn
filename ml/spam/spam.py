"""Run the Spam Classification Pipeline."""
from ml.spam.email_parser import EmailParser
from sklearn.feature_extraction.text import CountVectorizer
import glob


class SpamPipeline:
    def __init__(self):
        """Construct pipeline."""

    def execute_pipeline(self):
        file_list = glob.glob("data/spam/*")
        email_list = []
        for file in file_list:
            email_parser = EmailParser(file)
            print (f"{file}:  {len(email_parser.get_final_token_list())}")
            email_list.append(" ".join(email_parser.get_final_token_list()))
        
        print("Creating Vector Corpus")
        vectorizer = CountVectorizer()
        vectorizer.fit(email_list)

        X = vectorizer.transform(email_list)
        print(vectorizer.get_feature_names())
        print(X[0])