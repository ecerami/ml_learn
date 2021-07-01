wget http://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2
wget http://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2
wget http://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2
bzip2 -d *.tar.bz2
tar -xvf 20021010_spam.tar
tar -xvf 20021010_easy_ham.tar
tar -xvf 20021010_hard_ham.tar