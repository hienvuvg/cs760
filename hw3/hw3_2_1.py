# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from Evaluation import Evaluation

import os
from os import path

input_loc = path.join(path.dirname(__file__)) # group_er
df = pd.read_csv(input_loc +'/data/D2z.txt', delimiter=' ',header = None)  # Also read the first row

eval = Evaluation()

dataset = df.to_numpy()
print("Data size: ",np.shape(dataset))

X_train = dataset[:,0:-1] 
y_train = dataset[:,-1] 

# As mentioned by Tzu-Heng Huang on piazza: "Only need to implement logistic regression from scratch. You can use packages for knn questions."
# Configuration: default is “minkowski”, which results in the standard Euclidean distance when p = 2
knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski') 
knn.fit(X_train, y_train)

# %%
n_points = 41
min_range = -2
max_range = 2.1
X1 = np.arange(min_range,max_range, (max_range-min_range)/n_points)
X2 = np.arange(min_range,max_range, (max_range-min_range)/n_points)

# Create sample space
X_test = np.zeros((n_points**2, 2)) # two columns
i_begin = 0
for x_i in X1:
    for j in range(n_points):
        X_test[i_begin+j, :] = np.array([x_i, X2[j]]).T
    i_begin = i_begin + n_points


# Calculate the decision boundary
y_pred = knn.predict(X_test) 


# Plot the train data with a different color for each class
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

markers = ["+" , "v" , "s" , "<", ">"]
group = y_train
k = 1
for g in np.unique(group):
    i = np.where(group != g)
    ax2.scatter(X_train[i, 0], X_train[i, 1], label=g, s=10, alpha=1, marker=markers[k])
    k = k + 1


# Plot the decision boundary using the test data
group = y_pred
for g in np.unique(group):
    i = np.where(group != g)
    ax2.scatter(X_test[i, 0], X_test[i, 1], label=g, s=2, alpha=1)


# Format the plot
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.savefig(input_loc + '/figures/hw3_2_1.pdf', format='pdf', bbox_inches='tight') # Must call before show()
plt.show()

# %%
