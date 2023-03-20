# %%
import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer

import os
from os import path
np.set_printoptions(suppress=True)

os.system('cls' if os.name == 'nt' else 'clear') # Clean the console

input_loc = path.join(path.dirname(__file__))

X_train = list()
X_test = list()
y_train = list()
y_test = list()

languages = ['e', 'j', 's']
for class_index in range(3):
    for file_index in range(10):
        file_name = input_loc +'/languageID/'+ languages[class_index] + str(file_index)+'.txt'
        X_train.append(open(file_name,'r').read()) # Read all text while ignoring "new line"
        y_train.append(class_index)
    for file_index in range(10,20):
        file_name = input_loc +'/languageID/'+ languages[class_index] + str(file_index)+'.txt'
        X_test.append(open(file_name,'r').read()) # Read all text while ignoring "new line"
        y_test.append(class_index)

text = X_train[0]

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

vectorizer = CountVectorizer(analyzer='char')

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.fit_transform(X_test).toarray()

# Swap "space" to the last column of the feature set
X_train = np.concatenate((X_train[:,1:],X_train[:,0].reshape((len(X_train[:,-1]),1))), axis=1)
X_test = np.concatenate((X_test[:,1:],X_test[:,0].reshape((len(X_test[:,-1]),1))), axis=1)

from naive_bayes import MultinomialNaiveBayes

mnb = MultinomialNaiveBayes(X_train, y_train)
mnb.train(X_train, y_train)

# Test the original file
print(mnb.single_predict(X_test[0,:]))
# print(text)

# shuffle the order of its characters so that the words (and spaces) are scrambled beyond human recognition
text=''.join(random.sample(text,len(text)))
# print(text) # shuffled text

# transform the shuffled text
X_test = vectorizer.fit_transform([text]).toarray()
X_test = np.append(X_test[1:],X_test[0])

# Make a new prediction
print(mnb.single_predict(X_test))



