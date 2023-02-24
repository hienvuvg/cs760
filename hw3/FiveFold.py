# %%
import numpy as np
import pandas as pd

import os
from os import path

input_loc = path.join(path.dirname(__file__)) # group_er
df = pd.read_csv(input_loc +'/data/emails.csv')  # Also read the first row

mydf = df.iloc[:,1:] # drop the first column
# print(mydf.head())

dataset = mydf.to_numpy()
print("Data size: ",np.shape(dataset))

X = dataset[:,1:-1] 
y = dataset[:,-1] 

# %%
group_1 = dataset[   0:1000,1:]
group_2 = dataset[1000:2000,1:]
group_3 = dataset[2000:3000,1:]
group_4 = dataset[3000:4000,1:]
group_5 = dataset[4000:5000,1:]

train_set_1 = np.concatenate((group_2, group_3, group_4, group_5), axis=0)
train_set_2 = np.concatenate((group_1, group_3, group_4, group_5), axis=0)
train_set_3 = np.concatenate((group_1, group_2, group_4, group_5), axis=0)
train_set_4 = np.concatenate((group_1, group_2, group_3, group_5), axis=0)
train_set_5 = np.concatenate((group_1, group_2, group_3, group_4), axis=0)
train_set = [train_set_1, train_set_2, train_set_3, train_set_4, train_set_5]

test_set_1 = group_1
test_set_2 = group_2
test_set_3 = group_3
test_set_4 = group_4
test_set_5 = group_5
test_set = [test_set_1, test_set_2, test_set_3, test_set_4, test_set_5]


# for d_set in test_set:
#     print(np.shape(d_set))

# %%
