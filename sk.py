import re
import os
import os.path as path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import Stemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import multiprocessing
from functools import partial
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import WordNetLemmatizer

Enrons = 6
Processes = multiprocessing.cpu_count()
IsLemma = False

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
    def use_lemma(self, is_lemma):
        self.is_lemma = is_lemma
        self.wnl = WordNetLemmatizer()
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        if self.is_lemma:
            return lambda doc: [self.wnl.lemmatize(t) for t in analyzer(doc)]
        else:
            return lambda doc: Stemmer.Stemmer('en').stemWords(analyzer(doc))

class StemmedCountVectorizer(CountVectorizer):
    def use_lemma(self, is_lemma):
        self.is_lemma = is_lemma
        self.wnl = WordNetLemmatizer()
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        if self.is_lemma:
            return lambda doc: [self.wnl.lemmatize(t) for t in analyzer(doc)]
        else:
            return lambda doc: Stemmer.Stemmer('en').stemWords(analyzer(doc))

class MutualInformation:
    def __init__(self, analyzer='word', stop_words='english', max_df=1.0):
        self.vec = StemmedCountVectorizer(analyzer=analyzer, stop_words=stop_words, max_df=max_df)
    def fit_transform(self, raw_documents, labels):
        X = self.vec.fit_transform(raw_documents)
        A = X*labels
        B = X*(labels==0)
        C = None
        return X

def fold(feats, labels):
    length = labels.shape[0]
    iloc = np.random.permutation(length)
    _feats, _labels = feats[iloc], labels[iloc]
    c_test = int(round(length*10./100))
    return _feats[c_test:], _labels[c_test:], _feats[:c_test], _labels[:c_test]

def pickling(rep='tfidf', is_lemma=False):
    # rep='binary'
    contents = pickle.load(open('contents.pkl', 'r'))
    for max_df in np.arange(1., .0, -0.1):
        if rep=='tfidf':
            vec = StemmedTfidfVectorizer(analyzer='word', stop_words='english', max_df=max_df)
        if rep=='binary':
            vec = StemmedCountVectorizer(analyzer='word', stop_words='english', max_df=max_df, binary=True)
        vec.use_lemma(is_lemma)
        tfidf = vec.fit_transform(contents)
        print 'max_df {} shape {}'.format(max_df, tfidf.shape[1])
        with open('{}.{}.{}.pkl'.format(rep, 'lemma' if is_lemma else 'stem', max_df), 'w') as f:
            pickle.dump(tfidf, f)

def get_accuracy(name='tfidf', max_df=1.0, n_iter=4, clf=MultinomialNB(), use_lsa=False):
    # SVM: clf=SGDClassifier(loss='hinge', penalty='l2')
    labels = pickle.load(open('labels.pkl', 'r'))
    samples = pickle.load(open('{}.{}.pkl'.format(name, max_df), 'r'))
    if use_lsa:
        svd = TruncatedSVD(n_components=100)
        feats = svd.fit_transform(samples)
        if type(clf) == MultinomialNB:
            scaler = MinMaxScaler()
            feats = scaler.fit_transform(feats)
    else:
        feats = samples
    p_accs = .0
    for i in range(0, n_iter):
        feats_train, labels_train, feats_test, labels_test = fold(feats, labels)
        clf.fit(feats_train, labels_train)
        predicted = clf.predict(feats_test)
        c_accs = ((labels_test-predicted)==0).nonzero()[0].shape[0]
        p_accs += c_accs*100./labels_test.shape[0]
    return p_accs/n_iter

def count_features(name='tfidf', max_df=1.0):
    samples = pickle.load(open('{}.{}.pkl'.format(name, max_df), 'r'))
    return samples.shape[1]
