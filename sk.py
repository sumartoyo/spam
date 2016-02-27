import os
import os.path as path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import Stemmer
from sklearn.naive_bayes import MultinomialNB
import multiprocessing
from functools import partial
import pickle

Enrons = 6
MaxDf = 0.6
Processes = multiprocessing.cpu_count()

def read_file(path, re_nonletter, re_oneletter):
    with open(path, 'r') as f:
        content = f.read()
    content = re_nonletter.sub('', content)
    content = re_oneletter.sub(' ', content)
    return content

re_nonletter = re.compile(r'[^A-Za-z\s]+')
re_oneletter = re.compile(r'\s+[A-Za-z]{1}\s+')
partial_read_file = partial(read_file, re_nonletter=re_nonletter, re_oneletter=re_oneletter)

def get_enron():

    enrons = [path.join('..', 'enron', 'enron{}'.format(i+1)) for i in range(0, Enrons)]
    files, labels = [], []
    for cat in [('spam', 1), ('ham', 0)]:
        dirs = [path.join(e, cat[0]) for e in enrons]
        fs = [path.join(d, f) for d in dirs for f in os.listdir(d) if path.isfile(path.join(d, f))]
        labels.extend([cat[1]] * len(fs))
        files.extend(fs)

    pool = multiprocessing.Pool(processes=Processes)
    contents = pool.map(partial_read_file, files)
    pool.close()
    pool.join()

    return pd.DataFrame({'contents':contents, 'labels':labels})

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: Stemmer.Stemmer('en').stemWords(analyzer(doc))

def train(samples, labels):
    clf = MultinomialNB()
    clf.fit(samples, labels)
    return clf

def robot():

    print 'read'
    # df = get_enron()
    contents, labels = pickle.load(open('contents.pkl', 'r')), pickle.load(open('labels.pkl', 'r'))

    print 'fold'
    # because KFold is too mainstream
    df = df.iloc[np.random.permutation(len(df))]
    c_test = int(round(len(df)*10./100))
    df_train, df_test = df[c_test:], df[:c_test]
    contents_train, labels_train = df_train['contents'].values, df_train['labels'].values
    contents_test, labels_test = df_test['contents'].values, df_test['labels'].values

    print 'tfidf'
    tfidf = StemmedTfidfVectorizer(analyzer='word', stop_words='english', max_df=MaxDf)
    features_train = tfidf.fit_transform(contents_train)
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
