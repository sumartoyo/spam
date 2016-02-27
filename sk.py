import re
import os
import os.path as path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import Stemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import multiprocessing
from functools import partial
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

Enrons = 6
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

def fold(feats, labels):
    length = labels.shape[0]
    iloc = np.random.permutation(length)
    _feats, _labels = feats[iloc], labels[iloc]
    c_test = int(round(length*10./100))
    return _feats[c_test:], _labels[c_test:], _feats[:c_test], _labels[:c_test]

def pickling():
    contents = pickle.load(open('contents.pkl', 'r'))
    for max_df in np.arange(1., .0, -0.1):
        vec = StemmedTfidfVectorizer(analyzer='word', stop_words='english', max_df=max_df)
        tfidf = vec.fit_transform(contents)
        print 'max_df {} shape {}'.format(max_df, tfidf.shape[1])
        with open('tfidf.{}.pkl'.format(max_df), 'w') as f:
            pickle.dump(tfidf, f)

def get_accuracy(max_df, n_iter=4, clf=MultinomialNB(), use_lsa=False):
    # SVM: clf=SGDClassifier(loss='hinge', penalty='l2')
    labels = pickle.load(open('labels.pkl', 'r'))
    tfidf = pickle.load(open('tfidf.{}.pkl'.format(max_df), 'r'))
    if use_lsa:
        svd = TruncatedSVD(n_components=100)
        feats = svd.fit_transform(tfidf)
        if type(clf) == MultinomialNB:
            scaler = MinMaxScaler()
            feats = scaler.fit_transform(feats)
    else:
        feats = tfidf
    p_accs = .0
    for i in range(0, n_iter):
        feats_train, labels_train, feats_test, labels_test = fold(feats, labels)
        clf.fit(feats_train, labels_train)
        predicted = clf.predict(feats_test)
        c_accs = ((labels_test-predicted)==0).nonzero()[0].shape[0]
        p_accs += c_accs*100./labels_test.shape[0]
    return p_accs/n_iter
