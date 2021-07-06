import email
import pandas as pd
from ml.spam.email_parser import EmailParser


def test_email_parser():
    email_parser = EmailParser("tests/data/spam0.txt")
    tokens = email_parser.get_final_token_list()
    assert len(tokens) == 97

    email_parser = EmailParser("tests/data/spam1.txt")
    tokens = email_parser.get_final_token_list()
    assert len(tokens) == 157
