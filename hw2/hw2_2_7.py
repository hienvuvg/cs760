
# %%
import numpy as np
import pandas as pd
import os
from os import path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

class TreeNode():
    def __init__(self, feature_index=None, split_threshold=None, left_group=None, right_group=None, gain_ratio=None, node_class=None):
        self.feature_index = feature_index
        self.gain_ratio = gain_ratio
        self.split_threshold = split_threshold
        self.left_group = left_group
        self.right_group = right_group
        self.node_class = node_class

class DecisionTree():
    def __init__(self):
        self.root = None
        self.n_node = 0
    
    def Entropy(self, samples):
        unique_values = np.unique(samples)
        entropy_value = 0
        for value in unique_values:
            value_ratio = len(samples[samples == value]) / len(samples)
            entropy_value -= value_ratio * np.log2(value_ratio)
        return entropy_value

    def InfoGain(self, values, left_values, right_values):
        left_weight = len(left_values) / len(values)
        right_weight = len(right_values) / len(values)
        info_gain = self.Entropy(values) - (left_weight*self.Entropy(left_values) + right_weight*self.Entropy(right_values))
        return info_gain

    def GainRatio(self, values, left_values, right_values):
        left_weight = len(left_values) / len(values)
        right_weight = len(right_values) / len(values)
        intrinsic_info = - left_weight*np.log2(left_weight) - right_weight*np.log2(right_weight)
        gain_ratio = self.InfoGain(values, left_values, right_values) / intrinsic_info
        return gain_ratio

    def SplitNode(self, dataset, feature_index, split_threshold):
        left_group = list()
        right_group = list()

        # Do not sort the dataset by one feature
        for data_sample in dataset:
            if data_sample[feature_index] >= split_threshold:
                left_group.append(data_sample)
            else:
                right_group.append(data_sample)

        left_group = np.asarray(left_group)
        right_group = np.asarray(right_group)
        return left_group, right_group

    def FindBestSplit(self, dataset, n_features):
        best_split = {}
        max_gain_ratio = -1
        
        # check each feature
        for feature_index in range(n_features):
            single_feature_set = dataset[:, feature_index]
            threshold_candidates = np.unique(single_feature_set) # use unique values as potential candidate splits

            for split_threshold in threshold_candidates:
                left_group, right_group = self.SplitNode(dataset, feature_index, split_threshold)

                # splitting when both groups are non-empty
                if len(left_group)>0 and len(right_group)>0:
                    values, left_values, right_values = dataset[:, -1], left_group[:, -1], right_group[:, -1]
                    gain_ratio = self.GainRatio(values, left_values, right_values)
                    left_entropy = self.Entropy(left_values)
                    right_entropy = self.Entropy(right_values)

                    # # assign gain_rato to zero to stop splitting when the entropy of either left or right group is zero
                    if left_entropy == 0 or right_entropy == 0:
                        gain_ratio = 0

                    # Store the best split
                    if gain_ratio > max_gain_ratio:
                        best_split["feature_index"] = feature_index
                        best_split["split_threshold"] = split_threshold
                        best_split["left_group"] = left_group
                        best_split["right_group"] = right_group
                        best_split["gain_ratio"] = gain_ratio
                        # update max value
                        max_gain_ratio = gain_ratio
                        
        return best_split 


    def FindNodeClass(self, class_values):
        class_values = list(class_values)
        majority_class = max(class_values, key=class_values.count)
        return majority_class


    def MakeSubTree(self, data): 
        features, values = data[:,:-1], data[:,-1]
        n_samples, n_features = np.shape(features)

        # counting number of nodes
        self.n_node = self.n_node + 1
        
        # only split when at lease two samples left, stop when it's empty
        if n_samples > 1:
            best_split = self.FindBestSplit(data, n_features)
           
            if best_split["gain_ratio"] > 0: # stop if gain ratio is zero
                left_tree = self.MakeSubTree(best_split["left_group"])
                right_tree = self.MakeSubTree(best_split["right_group"])
                return TreeNode(best_split["feature_index"], best_split["split_threshold"], left_tree, right_tree, best_split["gain_ratio"]) # Return new node
        
        node_class = self.FindNodeClass(values)
        leaf = TreeNode(node_class = node_class) # Make a leaf
        return leaf


    def BuildTree(self, X_train, y_train):
        y_train = np.reshape(y_train, (len(y_train), 1))
        dataset = np.concatenate((X_train, y_train), axis=1)
        self.root = self.MakeSubTree(dataset)


    # For the prediction -------

    def Prediction(self, X_test, DT):
        # Stop recursion when reaching a leaf
        if DT.node_class!=None: 
            return DT.node_class

        # Evaluating X_test using the tree
        if X_test[DT.feature_index] >= DT.split_threshold:
            return self.Prediction(X_test, DT.left_group)
        else:
            return self.Prediction(X_test, DT.right_group)


    def PredictTree(self, X_test):
        y_pred = list()
        for data_sample in X_test:
            y = self.Prediction(data_sample, self.root)
            y_pred.append(y)

        return np.asarray(y_pred)

    def n_Node(self):
        return self.n_node


# Main from here ----------------------------

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
    print(train_size)

    cls = DecisionTree()
    cls.BuildTree(X_train, y_train)

    print('Number of nodes: ', cls.n_Node())

    y_pred = cls.PredictTree(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    err_n = 1 - accuracy
    # print('Test accuray: ', round(accuracy, 3))
    print('Test error rate: ', round(err_n, 3))
    print(' ')

    err_array = np.append(err_array, err_n)
    train_array = np.append(train_array, train_size)

    # Visualization -------------
    n_points = 500
    min_x = -1.6
    max_x = 1.6
    X1 = np.arange(min_x,max_x,(max_x-min_x)/n_points)
    X2 = np.arange(min_x,max_x,(max_x-min_x)/n_points)
    pos = np.zeros((n_points,n_points))

    x1 = np.reshape(X1, (len(X1), 1))
    x2 = np.reshape(X2, (len(X2), 1))

    X_t = np.zeros((n_points, 2))
    for i in range(n_points):
        for j in range(n_points):
            X_t[j,:] = np.array([x1[i], x2[j]]).T
        pos[:,i] = cls.PredictTree(X_t) 

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    group = y
    for g in np.unique(group):
        i = np.where(group != g)
        ax2.scatter(X[i, 0], X[i, 1], label=g, s=3, alpha=1)

    ax2.contourf(X1, X2, pos, cmap='Spectral', alpha=0.5)
    # plt.axis('equal')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.savefig(input_loc + '/figures/hw2_2_7_'+str(train_size)+'.pdf', format='pdf', bbox_inches='tight') # Must call before show()


plt.figure(figsize=(5, 3.2), dpi=100)
plt.plot(train_array[1:], err_array[1:])
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.xlabel('Training size')
plt.ylabel('Test error rate')
plt.xlim([0,9000])
plt.tight_layout()
plt.savefig(input_loc + '/figures/hw2_2_7.pdf', format='pdf', bbox_inches='tight') # Must call before show()
plt.show()


# %%
