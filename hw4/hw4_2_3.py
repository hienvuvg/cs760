# %%
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import os
from os import path

os.system('cls' if os.name == 'nt' else 'clear') # Clean the console
np.set_printoptions(suppress=True)

input_loc = path.join(path.dirname(__file__))

X_train = list()
X_test = list()
y_train = list()
y_test = list()

languages = ['e', 'j', 's']
for i in range(3):
    for j in range(10):
        file_name = input_loc +'/languageID/'+ languages[i] + str(j)+'.txt'
        X_train.append(open(file_name,'r').read()) # Read all text while ignoring "space"
        y_train.append(i)
    for j in range(10,20):
        file_name = input_loc +'/languageID/'+ languages[i] + str(j)+'.txt'
        X_test.append(open(file_name,'r').read()) # Read all text while ignoring "space"
        y_test.append(i)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

vectorizer = CountVectorizer(analyzer='char')

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.fit_transform(X_test).toarray()

# Swap "space" to the last column of the feature set
X_train = np.concatenate((X_train[:,1:],X_train[:,0].reshape((len(X_train[:,-1]),1))), axis=1)
X_test = np.concatenate((X_test[:,1:],X_test[:,0].reshape((len(X_test[:,-1]),1))), axis=1)


from naive_bayes import MultinomialNaiveBayes

alpha = 0.5
mnb = MultinomialNaiveBayes(X_train, y_train, alpha)


prior_proba = mnb.train(X_train, y_train)
print('Conditional probability for Japanese:\n', np.round(mnb.conditional_proba[:,1],3))
print('Conditional probability for Spainsh:\n', np.round(mnb.conditional_proba[:,2],3))


