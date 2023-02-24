# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from Evaluation import Evaluation
from FiveFold import * 

# %%
eval = Evaluation()

i = 0
for train_data, test_data in zip(train_set, test_set):
	i = i + 1

	# This makes no sense:
	# scaler = MinMaxScaler()
	# X_train = scaler.fit_transform(train_data[:,0:-1])
	# X_test = scaler.transform(test_data[:,0:-1])

	X_train = train_data[:,0:-1]
	X_test = test_data[:,0:-1]
	y_train = train_data[:,-1]
	y_test = test_data[:,-1]

	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(X_train, y_train)

	y_pred = knn.predict(test_data[:,0:-1])

	print('\nFold', i)
	print('Accuracy :', round(eval.Accuracy(y_test, y_pred), 3))
	print('Precision:', round(eval.Precision(y_test, y_pred), 3))
	print('Recall   :', round(eval.Recall(y_test, y_pred), 3))

	# # For comparison only
	# from sklearn.metrics import accuracy_score
	# from sklearn.metrics import precision_score
	# from sklearn.metrics import recall_score
	# print('Precision: ', precision_score(y_test, y_pred))
	# print('Accuracy: ', accuracy_score(y_test, y_pred))
	# print('Recall: ', recall_score(y_test, y_pred))

# %%
