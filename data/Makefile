# Get the Titanic Data Set
# See:  https://www.kaggle.com/c/titanic/data
get_titanic:
	kaggle competitions download -c titanic
	unzip -d titanic titanic.zip

# Get the MNIST Digit Data Set
get_mnist:
	wget https://www.openml.org/data/get_csv/52667/mnist_784.arff
	mv mnist_784.arff mnist_784.csv

# Get the SPAM Data Set
get_spam:
	wget http://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2
	wget http://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2
	bzip2 -d *.tar.bz2
	tar -xvf 20021010_spam.tar
	tar -xvf 20021010_easy_ham.tar

# Get the Kaggle Twitter NLP Data Set
# See:  https://www.kaggle.com/c/nlp-getting-started/data
get_twitter_nlp:
	kaggle competitions download -c nlp-getting-started
	unzip -d twitter nlp-getting-started.zip

clean:
	rm -f *.csv
	rm -f *.tar*
	rm -f *.zip
	rm -f *.arff
	rm -rf twitter
	rm -rf spam
	rm -rf easy_ham
	rm -rf titanic
