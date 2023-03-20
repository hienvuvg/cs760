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

mnb.train(X_train, y_train)

# for row in X_test:
y_hat = np.zeros(mnb.n_classes)
for i, y_class in enumerate(mnb.class_values): # Go through each class
    y_hat[i] = mnb.likelihood(X_test[0], y_class)   # Calculate the log probability for each class, i.e. posterior p_hat(y_i|x) 
    # y_hat[i].append(np.argmax(y_hat[i])) # Adding the only class with maximum likelihood to the list
# print(y_hat, end=" ")
# print(np.argmax(y_hat))

print('p0:\n', np.round(y_hat[0]))
print('p1:\n', np.round(y_hat[1]))
print('p2:\n', np.round(y_hat[2]))


