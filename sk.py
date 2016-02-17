import os
import os.path as path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.lancaster import LancasterStemmer
from sklearn import svm
import threading
import multiprocessing

Stemmer = LancasterStemmer()
RgxNonletter = re.compile(r'[^A-Za-z\s]+')
RgxOnechar = re.compile(r'\s+[A-Za-z]{1}\s+')
Dimension = 2000
MinIdf = 4
Cpus = multiprocessing.cpu_count()

def read_doc(file):
    f = open(file)
    content = f.read()
    f.close()
    return content

def clean(contents):
    for i in range(0, len(contents)):
        content = contents[i]
        content = RgxNonletter.sub('', content)
        content = RgxOnechar.sub(' ', content)
        contents[i] = content

def read_dir(dir, limit=None):
    contents = []
    for doc in [f for f in os.listdir(dir) if path.isfile(path.join(dir, f))]:
        content = read_doc(path.join(dir, doc))
        contents.append(content)
    clean(contents)
    if limit: # DEBUG
        contents[:] = contents[:limit]
    return contents

def get_enron():
    c_enron = 4
    contents, labels = [], []
    dir_enron = path.join('..', 'enron')
    for i_enron in range(0, c_enron):
        dir_version = path.join(dir_enron, 'enron{}'.format(i_enron+1))
        for label in [0, 1]:
            cat = 'ham' if label == 0 else 'spam' if label == 1 else None
            dir_cat = path.join(dir_version, cat)
            limit = 1000 if label == 0 else 500 if label == 1 else None
            contents_dir = read_dir(dir_cat, limit)
            contents.extend(contents_dir)
            labels.extend([label] * len(contents_dir))
    return contents, labels

def get_tokens(content):
    tokens = nltk.word_tokenize(content)
    stems = [Stemmer.stem(token) for token in tokens]
    return stems

def get_tfidf(contents):
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', tokenizer=get_tokens)
    tfidf = vectorizer.fit_transform(contents)
    return tfidf, vectorizer

def select(idf):
    argsort = np.argsort(idf)
    i = (idf < MinIdf).nonzero()[0].shape[0]
    j = i + Dimension
    len_idf = idf.shape[0]
    j = len_idf if j > len_idf else j
    return argsort, i, j

def train(samples, labels):
    clf = svm.SVC()
    clf.fit(samples, labels)
    return clf

def transform_tfidf(contents, transformer, argsort, i, j):
    tfidf = transformer(contents)
    return tfidf[:,argsort][:,i:j]

def robot():
    
    print 'fetch'
    contents, labels = get_enron()
    c_spams = np.array(labels).nonzero()[0].shape[0]
    print 'spam:ham {}:{}'.format(c_spams, len(labels)-c_spams)
    
    print 'tfidf'
    tfidf, vectorizer = get_tfidf(contents)

    # print 'select'
    # idf = vectorizer.idf_
    # argsort, i, j = select(idf)
    # tfidf = tfidf[:,argsort][:,i:j]

    print 'train'
    clf = train(tfidf, labels)
    
    def test(label):
        cat = 'ham' if label == 0 else 'spam' if label == 1 else None
        dir = path.join('..', 'enron', 'enron6', cat)
        contents_test = read_dir(dir)
        # tfidf_test = transform_tfidf(contents_test, vectorizer.transform, argsort, i, j)
        tfidf_test = vectorizer.transform(contents_test)
        c_contents = len(contents_test)
        print '{} count'.format(cat), c_contents
        classified = clf.predict(tfidf_test)
        c_positives = classified.nonzero()[0].shape[0]
        print '{} error'.format(cat), c_positives*100./c_contents if label == 1 else (c_contents-c_positives)*100./c_contents if label == 0 else None

    print 'test'
    threads = [threading.Thread(target=test, args=(label,)) for label in [0, 1]]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
