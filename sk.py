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
import multiprocessing
from multiprocessing.pool import ThreadPool

Stemmer = PorterStemmer()
MinIdf = 4
Ccpus = multiprocessing.cpu_count()

def get_enron():

    c_enron = 6
    enrons = [path.join('..', 'enron', 'enron{}'.format(i+1)) for i in range(0, c_enron)]
    files, labels = [], []
    for cat in [('spam', 1), ('ham', 0)]:
        dirs = [path.join(e, cat[0]) for e in enrons]
        fs = [path.join(d, f) for d in dirs for f in os.listdir(d) if path.isfile(path.join(d, f))]
        labels.extend([cat[1]] * len(fs))
        files.extend(fs)
    print '  got files and labels'
    
    re_nonletter = re.compile(r'[^A-Za-z\s]+')
    re_oneletter = re.compile(r'\s+[A-Za-z]{1}\s+')
    def read_files(path):
        f = open(path)
        content = f.read()
        f.close()
        content = re_nonletter.sub('', content)
        content = re_oneletter.sub(' ', content)
        return content
    pool = ThreadPool(processes=Ccpus)
    contents = pool.map(read_files, files)
    print '  read and cleaned'

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
        # self.stop = self.start + dimension
        # if self.stop > self.vectorizer.idf_.shape[0]:
        #     self.stop = self.vectorizer.idf_.shape[0]

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
    
    # print 'tfidf'
    # tfidf = Tfidf()
    # features_train = tfidf.fit(contents_train)
    # print '  features_train', features_train.shape

    # print 'train'
    # clf = train(features_train, labels_train)

    # print 'test'
    # features_test = tfidf.transform(contents_test)
    # labels_predict = clf.predict(features_test)
    # diffs = labels_test - labels_predict
    # false_spam = (diffs == 1).nonzero()[0].shape[0]
    # false_ham = (diffs == -1).nonzero()[0].shape[0]
    # c_spam = labels_test.nonzero()[0].shape[0]
    # c_ham = labels_test.shape[0] - c_spam
    # print '  harusnya spam tapi bukan {}'.format(false_spam*100./c_spam)
    # print '  harusnya ham tapi bukan {}'.format(false_ham*100./c_ham)
