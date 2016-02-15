import os
import os.path as path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.lancaster import LancasterStemmer
from sklearn import svm
import threading
import itertools

stemmer = LancasterStemmer()
rgx_nonletter = re.compile(r'[^A-Za-z\s]+')
rgx_onechar = re.compile(r'\s+[A-Za-z]{1}\s+')
n_features = 2000
min_idf = 4

def read_doc(file):
    f = open(file)
    content = f.read()
    f.close()
    content = rgx_nonletter.sub('', content)
    content = rgx_onechar.sub(' ', content)
    return content

def get_contents(dir):
    contents = []
    for doc in [f for f in os.listdir(dir) if path.isfile(path.join(dir, f))]:
        content = read_doc(path.join(dir, doc))
        contents.append(content)
    return contents

def get_tokens(content):
    tokens = nltk.word_tokenize(content)
    stems = [stemmer.stem(token) for token in tokens]
    return stems

def train():

    def read_dir(dir, label, contents_dir, labels_dir):
        contents_dir[:] = get_contents(dir)
        labels_dir[:] = [label] * len(contents_dir)

    c_enron = 1
    contents, labels = [], []
    dir = path.join('..', 'dimas')
    for i in range(0, c_enron):
        ver = path.join(dir, 'enron{}'.format(i+1))
        contents_s, labels_s = [], []
        contents_h, labels_h = [], []
        thread_s = threading.Thread(target=read_dir, args=(path.join(ver, 'spam'), 1, contents_s, labels_s))
        thread_h = threading.Thread(target=read_dir, args=(path.join(ver, 'ham'), 0, contents_h, labels_h))
        thread_s.start()
        thread_h.start()
        thread_s.join()
        thread_h.join()
        contents.extend(contents_s)
        labels.extend(labels_s)
        contents.extend(contents_h)
        labels.extend(labels_h)

    print '\tendof dir'

    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', tokenizer=get_tokens)
    tfidf = vectorizer.fit_transform(contents)
    idf = vectorizer.idf_
    words = vectorizer.get_feature_names()

    print '\tendof tfidf'

    argsort = np.argsort(idf)
    i_start = (idf < min_idf).nonzero()[0].shape[0]
    i_end = i_start + n_features
    tfidf_selected = tfidf[:,argsort]#[:,i_start:i_end]
    words_selected = np.array(words)#[argsort][i_start:i_end]
    features = np.zeros(tfidf_selected.shape, dtype=np.uint8)
    features[tfidf_selected.nonzero()] = 1

    print '\tendof select features'

    clf = svm.SVC()
    clf.fit(features, labels)

    print '\tendof train'

    return clf, words_selected

def get_feature(file, words):
    content = read_doc(file)
    tokens = set(get_tokens(content))
    feature = []
    for word in words:
        feature.append(1 if word in tokens else 0)
    return feature

def robot():
    
    print 'train'
    clf, words = train()
    
    def test(type, words):
        count = 0
        features = []
        dir = path.join('..', 'dimas', 'enron1', type)
        for doc in [f for f in os.listdir(dir) if path.isfile(path.join(dir, f))]:
            count += 1
            feature = get_feature(path.join(dir, doc), words)
            features.append(feature)
        print '{} count'.format(type), count
        classified = clf.predict(features)
        nonzero = classified.nonzero()[0].shape[0]
        print '{} nonzero'.format(type), 100.*nonzero/count

    print 'test'
    thread_s = threading.Thread(target=test, args=('spam', words))
    thread_h = threading.Thread(target=test, args=('ham', words))
    thread_s.start()
    thread_s.join()
    thread_h.start()
    thread_h.join()
