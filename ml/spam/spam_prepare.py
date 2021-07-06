"""Run the Spam Pre-Processing Pipeline."""
from ml.spam.email_parser import EmailParser
from sklearn.feature_extraction.text import CountVectorizer
import glob
from random import seed
from random import random
import progressbar


class SpamPreparePipeline:
    def __init__(self):
        """Construct pipeline."""
        seed(42)

    def execute_pipeline(self):
        train_file = "out/spam_train.txt"
        test_file = "out/spam_test.txt"
        train_fd = open(train_file, "w")
        test_fd = open(test_file, "w")

        self.process_email_dir("data/spam", 1, train_fd, test_fd)
        self.process_email_dir("data/easy_ham", 0, train_fd, test_fd)

        train_fd.close()
        test_fd.close()
        print(f"Training data written to:  {train_file}.")
        print(f"Test data written to:  {test_file}.")

    def process_email_dir(self, email_dir, spam_flag, train_fd, test_fd):
        file_list = glob.glob(email_dir + "/*")
        print(f"Pre-processing messages in {email_dir}: {len(file_list)}")
        for i in progressbar.progressbar(range(len(file_list))):
            current_file = file_list[i]
            email_parser = EmailParser(current_file)
            self.write_email_out(current_file, spam_flag, email_parser, train_fd, test_fd)

    def write_email_out(self, current_file, spam_flag, email_parser, train_fd, test_fd):
        r = random()
        token_str = " ".join(email_parser.get_final_token_list())
        out_str = f"{spam_flag}\t{current_file}\t{token_str}\n"
        if r < 0.3:
            test_fd.write(out_str)
        else:
            train_fd.write(out_str)
