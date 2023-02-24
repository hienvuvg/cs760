# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Evaluation import Evaluation
from LogisticRegression import LogisticRegression
from FiveFold import * 


learning_rate = 5
n_epochs = 1000

eval = Evaluation()

i = 0
for train_data, test_data in zip(train_set, test_set):
	i = i + 1

	X_train = train_data[:,0:-1]
	X_test = test_data[:,0:-1]
	y_train = train_data[:,-1]
	y_test = test_data[:,-1]

	n_features = X_train.shape[1]
	cls = LogisticRegression(n_features)
	loss = cls.fit(X_train, y_train, epochs=n_epochs, lr=learning_rate)
	y_pred = cls.predict(X_test)

	print('\nFold', i)
	print('Accuracy :', round(eval.Accuracy(y_test, y_pred), 3))
	print('Precision:', round(eval.Precision(y_test, y_pred), 3))
	print('Recall   :', round(eval.Recall(y_test, y_pred), 3))
    
# %%
