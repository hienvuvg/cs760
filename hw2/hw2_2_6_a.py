# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from os import path

input_loc = path.join(path.dirname(__file__)) # Folder
df = pd.read_csv(input_loc +'/data/D1.txt', delimiter=' ', header = None)  # Also read the first row

A = df.to_numpy() # Array

X = A[:,0:-1] # multi column selection (#0 to #10)
y = A[:,-1] # single column selection (#11)

# plt.figure(figsize=(5, 3.5), dpi=100)
# plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.4)
# plt.grid(color='gray', linestyle=':', linewidth=0.5)
fig, ax = plt.subplots()
group = y
for g in np.unique(group):
    i = np.where(group != g)
    ax.scatter(X[i, 0], X[i, 1], label=g, s=8, alpha=1)

# plt.xlim([0,1])
# plt.ylim([0,1])
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.title("Scatter plot of D1.txt")

plt.tight_layout()
plt.savefig(input_loc + '/figures/hw2_2_6_D1.pdf', format='pdf', bbox_inches='tight') # Must call before show()
plt.show()


# %%

input_loc = path.join(path.dirname(__file__)) # Folder
df = pd.read_csv(input_loc +'/data/D2.txt', delimiter=' ', header = None)  # Also read the first row

A = df.to_numpy() # Array

X = A[:,0:-1] # multi column selection (#0 to #10)
y = A[:,-1] # single column selection (#11)

# plt.figure(figsize=(5, 3.5), dpi=100)
# plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.4)
# plt.grid(color='gray', linestyle=':', linewidth=0.5)
fig, ax = plt.subplots()
group = y
for g in np.unique(group):
    i = np.where(group != g)
    ax.scatter(X[i, 0], X[i, 1], label=g, s=8, alpha=1)

# plt.xlim([0,1])
# plt.ylim([0,1])
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.title("Scatter plot of D2.txt")

plt.tight_layout()
plt.savefig(input_loc + '/figures/hw2_2_6_D2.pdf', format='pdf', bbox_inches='tight') # Must call before show()
plt.show()

# %%
