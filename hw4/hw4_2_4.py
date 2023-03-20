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

file_name = input_loc +'/languageID/e10.txt'
X_train.append(open(file_name,'r').read()) # Read all text while ignoring "space"

vectorizer = CountVectorizer(analyzer='char')

X_train = vectorizer.fit_transform(X_train).toarray()

bag_of_words = vectorizer.get_feature_names_out()
# print('Word 0:', bag_of_words[0])

# Swap "space" to the last column of the feature set
X_train = np.concatenate((X_train[:,1:],X_train[:,0].reshape((len(X_train[:,-1]),1))), axis=1)
bag = np.append(bag_of_words[1:], bag_of_words[0])

print('Bag-of-words:', bag)

# prior_proba = mnb.train(X_train, y_train)
print(X_train)

for i in range(len(bag)):
    print(str(bag[i])+': '+str(X_train[0,i]))


