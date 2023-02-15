
# %%
import numpy as np
import pandas as pd
import os
from os import path

class FindCandidates():
    def __init__(self):
        self.root = None
    
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

    def ListCandidates(self, X_train, y_train):
        y_train = np.reshape(y_train, (len(y_train), 1))
        dataset = np.concatenate((X_train, y_train), axis=1)
        n_features = len(X_train[0,:])
        
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
                    info_gain = self.InfoGain(values, left_values, right_values)
                    left_entropy = self.Entropy(left_values)
                    right_entropy = self.Entropy(right_values)
                    # print(round(gain_ratio, 4))
                    print('Candidate split: X%d=%.1f  Entropy1=%.3f  Entropy2=%.3f  InfoGain=%.3f  GainRatio=%.3f' % 
                    ((feature_index), split_threshold, left_entropy, right_entropy, info_gain, gain_ratio))
    

# Main from here ----------------------------

# load and prepare data
input_loc = path.join(path.dirname(__file__)) # Folder
data = pd.read_csv(input_loc +'/data/Druns.txt', delimiter=' ',header = None)  # Also read the first row

dataset = data.to_numpy()

X = dataset[:,0:-1] 
y = dataset[:,-1] 

cls = FindCandidates()
cls.ListCandidates(X,y)
