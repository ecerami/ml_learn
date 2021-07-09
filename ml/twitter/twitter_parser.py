"""Parses / Prepares Single Twitter Message."""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup


class TwitterParser:
    def __init__(self, msg):
        """Construct pipeline."""
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

        # Strip HTML via Beautiful Soup
        soup = BeautifulSoup(msg, "html.parser")
        msg = soup.get_text()

        # Remove Punctuation and Numbers
        msg = re.sub("[^A-Za-z]", " ", msg)

        # Covert to Lower Case
        msg = msg.lower()

        # Tokenize, and remove stop words, e.g. and, the, etc.
        tokens = word_tokenize(msg)
        non_stop_tokens = []
        for word in tokens:
            if word not in stopwords.words("english") and len(word) > 1:
                non_stop_tokens.append(word)

        # Stem all words, e.g. running --> run
        self.final_tokens = []
        stemmer = PorterStemmer()
        for token in non_stop_tokens:
            self.final_tokens.append(stemmer.stem(token))

    def get_final_token_list(self):
        """Get the final list of parsed tokens."""
        return self.final_tokens
