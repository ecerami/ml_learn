"""Parses / Prepares Single Email Message."""
import mailparser
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup


class EmailParser:
    def __init__(self, f):
        """Construct pipeline."""
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        mail = mailparser.parse_from_file(f)

        # Merge Subject and Body
        self.body = self._clean_body(mail.body)
        text = mail.subject + " " + self.body

        # Strip HTML via Beautiful Soup
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        # Remove Punctuation and Numbers
        text = re.sub("[^A-Za-z]", " ", text)

        # Covert to Lower Case
        text = text.lower()

        # Tokenize, and remove stop words, e.g. and, the, etc.
        tokens = word_tokenize(text)
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

    def _clean_body(self, body):
        """Clean the email body by removing MIME encodings."""
        lines = body.split("\n")
        clean_body = []
        for line in lines:
            if len(line) > 50 and " " not in line:
                pass
            else:
                clean_body.append(line)
        return "\n".join(clean_body)
