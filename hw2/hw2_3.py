# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

import os
from os import path

input_loc = path.join(path.dirname(__file__)) # Folder
df = pd.read_csv(input_loc +'/data/Dbig.txt', delimiter=' ', header = None)  # Also read the first row

A = df.to_numpy()

X = A[:,0:-1] 
y = A[:,-1] 

train_size_indices = [0.0032, 0.0128, 0.0512, 0.2048, 0.8192]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, train_size=0.8192,  stratify=y, shuffle=True) 

err_array = None
train_array = None

X_train = 0
y_train = 0
for t_size in train_size_indices:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=t_size,  stratify=y, shuffle=True) 
    train_size = len(X_train[:,0])
    print('Training set size: '+str(train_size))

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    # tree.plot_tree(clf, class_names=True)
    print('Number of nodes: ', round(clf.tree_.node_count, 3))

    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    err_n = 1 - accuracy
    # print('Test accuray: ', round(accuracy, 3))
    print('Test error rate: ', round(err_n, 3))
    print(' ')

    err_array = np.append(err_array, err_n)
    train_array = np.append(train_array, train_size)

plt.figure(figsize=(5, 3.2), dpi=100)
plt.plot(train_array[1:], err_array[1:])
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.xlabel('Training size')
plt.ylabel('Test error rate')
plt.xlim([0,9000])
plt.tight_layout()
plt.savefig(input_loc + '/figures/hw2_3.pdf', format='pdf', bbox_inches='tight') # Must call before show()
plt.show()

# %%

x = [[1, 2],
     [3, 4]]
y = [[5], [6]]
print(np.shape(y))
np.concatenate([x, y], axis=1)

# %%
