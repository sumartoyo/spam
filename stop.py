import os
import os.path as path
import numpy as np
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

st = LancasterStemmer()
stop = set(stopwords.words('english'))
stop.update(['`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '[', '{', ']', '}', '\\', '|', ';', ':', "'", '"', ',', '<', '.', '>', '/', '?'])

fdoc_flag, fdoc = set([]), {}
fwords = []

cdoc = 0
dirspam = '..\\enron1\\spam\\'
for doc in [f for f in os.listdir(dirspam) if path.isfile(path.join(dirspam, f))]:
    cdoc += 1
    f = open(path.join(dirspam, doc))
    text_raw = f.read()
    f.close()
    text = unicode(text_raw.lower(), errors='ignore')

    fword_flag, fword = set([]), {}

    cword = 0
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token not in stop:
            cword += 1
            if token not in fword_flag:
                fword_flag.add(token)
                fword[token] = 1.
                if token not in fdoc_flag:
                    fdoc_flag.add(token)
                    fdoc[token] = 1.
                else:
                    fdoc[token] += 1
            else:
                fword[token] += 1

    for word in fword:
        fword[word] /= cword
    fwords.append(fword)

for fword in fwords:
    for word in fword:
        fword[word] *= cdoc / fdoc[word]

print fwords[2]

# nfdoc_word = np.asarray(fdoc_word)
# nfdoc_count = np.asarray(fdoc_count)
# fsort = nfdoc_count.argsort()
# print nfdoc_word[fsort][-20:-1]
# print nfdoc_count[fsort][-20:-1]
# print nfdoc_word.shape

    # tags = nltk.pos_tag(tokens)
    # for tag in tags:
    #     if tag[1][:2] in ['CD', 'FW', 'JJ', 'NN', 'RB', 'UH', 'VB']:
    #         stemmed = st.stem(tag[0])
    #         if stemmed not in fdoc_word:
    #             fdoc_word.append(stemmed)
    #             fdoc_count.append(0)
    #         fdoc_i = fdoc_word.index(stemmed)
    #         fdoc_count[fdoc_i] += 1
