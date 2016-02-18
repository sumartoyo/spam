import os
import os.path as path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import threading
import multiprocessing

Stemmer = PorterStemmer()
RgxNonletter = re.compile(r'[^A-Za-z\s]+')
RgxOnechar = re.compile(r'\s+[A-Za-z]{1}\s+')
MinIdf = 4

def clean(contents):

    def clean_range(start, stop):
        for c in range(start, stop):
            content = contents[c]
            content = RgxNonletter.sub('', content)
            content = RgxOnechar.sub(' ', content)
            contents[c] = content

    cpus = multiprocessing.cpu_count()
    width = len(contents) / cpus
    ranges = [[start, start+width] for start in [i*width for i in range(0, cpus)]]
    ranges[-1][-1] = len(contents)
    threads = [threading.Thread(target=clean_range, args=tuple(r)) for r in ranges]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]

def read_dir(folder):

    def read_file(path):
        f = open(path)
        content = f.read()
        f.close()
        return content

    files = [f for f in os.listdir(folder) if path.isfile(path.join(folder, f))]
    contents = [read_file(path.join(folder, f)) for f in files]
    return contents

def get_enron():
    c_enron = 6
    contents, labels = [], []
    dirs = [path.join('..', 'enron', 'enron{}'.format(i+1)) for i in range(0, c_enron)]
    for label in [0, 1]:
        if label == 0: cat = 'ham'
        if label == 1: cat = 'spam'
        cts = [read_dir(path.join(d, cat)) for d in dirs]
        cts[:] = [c for ct in cts for c in ct]
        contents.extend(cts)
        labels.extend([label] * len(cts))
    clean(contents)
    return pd.DataFrame({'contents':contents, 'labels':labels})

class Tfidf:
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', tokenizer=self.get_tokens)

    def get_tokens(self, content):
        tokens = nltk.word_tokenize(content)
        stems = [Stemmer.stem(token) for token in tokens]
        return stems

    def fit(self, contents):
        tfidf = self.vectorizer.fit_transform(contents)
        self.sort(tfidf.shape[1])
        return self.select(tfidf)

    def sort(self, dimension):
        self.argsort = np.argsort(self.vectorizer.idf_)
        self.start = (self.vectorizer.idf_ < MinIdf).nonzero()[0].shape[0]
        self.stop = self.start + dimension
        if self.stop > self.vectorizer.idf_.shape[0]:
            self.stop = self.vectorizer.idf_.shape[0]

    def select(self, tfidf):
        return tfidf[:,self.argsort][:,self.start:]
        # return tfidf[:,self.argsort][:,self.start:self.stop]

    def transform(self, contents):
        tfidf = self.vectorizer.transform(contents)
        return self.select(tfidf)

def train(samples, labels):
    # clf = svm.SVC()
    clf = MultinomialNB()
    clf.fit(samples, labels)
    return clf

def robot():
    
    print 'read'
    df = get_enron()

    print 'random'
    df = df.iloc[np.random.permutation(len(df))]
    c_test = int(round(len(df)*10./100))
    df_train, df_test = df[c_test:], df[:c_test]
    contents_train, labels_train = df_train['contents'].values, df_train['labels'].values
    contents_test, labels_test = df_test['contents'].values, df_test['labels'].values
    
    print 'tfidf'
    tfidf = Tfidf()
    features_train = tfidf.fit(contents_train)
    print '  features_train', features_train.shape

    print 'train'
    clf = train(features_train, labels_train)

    print 'test'
    features_test = tfidf.transform(contents_test)
    labels_predict = clf.predict(features_test)
    diffs = labels_test - labels_predict
    false_spam = (diffs == 1).nonzero()[0].shape[0]
    false_ham = (diffs == -1).nonzero()[0].shape[0]
    c_spam = labels_test.nonzero()[0].shape[0]
    c_ham = labels_test.shape[0] - c_spam
    print '  harusnya spam tapi bukan {}'.format(false_spam*100./c_spam)
    print '  harusnya ham tapi bukan {}'.format(false_ham*100./c_ham)
