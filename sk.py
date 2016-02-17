import os
import os.path as path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.lancaster import LancasterStemmer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import threading
import multiprocessing

Stemmer = LancasterStemmer()
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
    c_enron = 2
    contents, labels = [], []
    dir_enron = path.join('..', 'enron')
    for i_enron in range(0, c_enron):
        dir_version = path.join(dir_enron, 'enron{}'.format(i_enron+1))
        for label in [0, 1]:
            cat = 'ham' if label == 0 else 'spam' if label == 1 else None
            dir_cat = path.join(dir_version, cat)
            contents_dir = read_dir(dir_cat)
            contents.extend(contents_dir)
            labels.extend([label] * len(contents_dir))
    clean(contents)
    return contents, labels

def get_tokens(content):
    tokens = nltk.word_tokenize(content)
    stems = [Stemmer.stem(token) for token in tokens]
    return stems

def get_tfidf(contents):
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', tokenizer=get_tokens)
    tfidf = vectorizer.fit_transform(contents)
    return tfidf, vectorizer

def select(idf, dimension):
    argsort = np.argsort(idf)
    start = (idf < MinIdf).nonzero()[0].shape[0]
    stop = start + dimension
    if stop > idf.shape[0]:
        stop = idf.shape[0]
    return argsort, start, stop

def train(samples, labels):
    # clf = svm.SVC()
    clf = MultinomialNB()
    clf.fit(samples, labels)
    return clf

def transform_tfidf(contents, transformer, argsort, start, stop):
    tfidf = transformer(contents)
    return tfidf[:,argsort][:,start:stop]

def robot():
    
    print 'read'
    contents, labels = get_enron()

    print 'random'
    df = pd.DataFrame({'contents':contents, 'labels':labels})
    df = df.iloc[np.random.permutation(len(labels))]
    c_test = int(round(len(labels)*10./100))
    df_train, df_test = df[c_test:], df[:c_test]
    contents_train, labels_train = df_train['contents'].values, df_train['labels'].values
    contents_test, labels_test = df_test['contents'].values, df_test['labels'].values
    
    print 'tfidf'
    features_train, vectorizer = get_tfidf(contents_train)

    print 'select'
    idf = vectorizer.idf_
    argsort, start, stop = select(idf, contents_train.shape[0])
    features_train = features_train[:,argsort][:,start:stop]
    print '  features_train', features_train.shape

    print 'train'
    clf = train(features_train, labels_train)

    print 'test'
    features_test = transform_tfidf(contents_test, vectorizer.transform, argsort, start, stop)
    labels_predict = clf.predict(features_test)
    diffs = labels_test - labels_predict
    false_spam = (diffs == 1).nonzero()[0].shape[0]
    false_ham = (diffs == -1).nonzero()[0].shape[0]
    c_spam = labels_test.nonzero()[0].shape[0]
    c_ham = labels_test.shape[0] - c_spam
    print '  harusnya spam tapi bukan {}'.format(false_spam*100./c_spam)
    print '  harusnya ham tapi bukan {}'.format(false_ham*100./c_ham)
