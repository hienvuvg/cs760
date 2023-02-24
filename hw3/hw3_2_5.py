# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from LogisticRegression import LogisticRegression
from Evaluation import Evaluation
from FiveFold import * 


eval = Evaluation()

# %% Data preparation
train_data = train_set_5 # From 1-4000
test_data = test_set_5 # From 4001-5000

X_train = train_data[:,0:-1]
X_test = test_data[:,0:-1]
y_train = train_data[:,-1]
y_test = test_data[:,-1]

# %% kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_proba_knn = knn.predict_proba(test_data[:,0:-1])

# %% Logistic regression
learning_rate = 5
n_epochs = 1000
n_features = X_train.shape[1]

cls = LogisticRegression(n_features)
loss = cls.fit(X_train, y_train, epochs=n_epochs, lr=learning_rate)
y_pred_proba_lg = cls.predict_proba(X_test)

# %% Plotting
plt.figure(figsize=(5, 3.5), dpi=100)

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_proba_knn[:,1])
auc = metrics.roc_auc_score(y_test, y_pred_proba_knn[:,1])
plt.plot(fpr,tpr,label="kNeighborsClassifier, auc="+str(round(auc,2)))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_proba_lg[:,1])
auc = metrics.roc_auc_score(y_test, y_pred_proba_lg[:,1])
plt.plot(fpr,tpr,label="LogisticRegression, auc="+str(round(auc,2)))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Analysis')
plt.legend()
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.savefig(input_loc + '/figures/hw3_2_5.pdf', format='pdf', bbox_inches='tight') # Must call before show()

plt.show()


# %%
