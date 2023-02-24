# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from Evaluation import Evaluation
from FiveFold import * 


eval = Evaluation()

n_neighbors = [1, 3, 5, 7, 10]
average_accuracy = list() 
for k in n_neighbors:
	i = 0
	accuracy = 0
	for train_data, test_data in zip(train_set, test_set):
		i = i + 1

		X_train = train_data[:,0:-1]
		X_test = test_data[:,0:-1]
		y_train = train_data[:,-1]
		y_test = test_data[:,-1]

		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)

		y_pred = knn.predict(test_data[:,0:-1])

		accuracy += eval.Accuracy(y_test, y_pred)
	average_accuracy.append(accuracy / len(n_neighbors))

# %%
print(average_accuracy)

plt.figure(figsize=(5, 3.5), dpi=100)
plt.plot(n_neighbors, average_accuracy, linewidth=2, marker='o')
plt.xlim([0,10])
plt.xlabel('k')
plt.ylabel('Average accuracy')
plt.title('kNN 5-Fold Cross Validation')
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.savefig(input_loc + '/figures/hw3_2_4.pdf', format='pdf', bbox_inches='tight') # Must call before show()
plt.show()

# %%
