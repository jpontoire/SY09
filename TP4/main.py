import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import add_labels
import scipy.linalg as linalg
from math import sqrt

# 1.
notes = pd.read_csv("data/notes.txt", sep="\\s+")
cls = PCA(n_components=5)
pcs = cls.fit_transform(notes)

# plt.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5"], cls.explained_variance_ratio_)
# plt.show()

# 2.
# plt.scatter(pcs[:, 0], pcs[:, 1])
# add_labels(pcs[:, 0], pcs[:, 1], notes.index)
# plt.show()

# 3.
# plt.scatter(pcs[:, 0], pcs[:, 1])
# add_labels(pcs[:, 0], pcs[:, 1], notes.index)

notes_2 = pd.DataFrame([[8.0, 6.0, 10.0, 9, 14], [10, 11, 4.5, 8.0, 6]], columns=notes.columns, index=["Alice", "Steve"])

pcs1 = cls.transform(notes_2)
# plt.scatter(pcs1[:, 0], pcs1[:, 1])
# add_labels(pcs1[:, 0], pcs1[:, 1], notes_2.index)
# plt.show()


# 4.
crabs = pd.read_csv("data/crabs.csv", sep=r"\s+")
crabsquant = crabs.iloc[:, 3:8]

# 5.
crabs_long = crabs.melt(id_vars=["sp", "sex", "index"])
# sns.boxplot(x="variable", y="value", data=crabs_long)
# plt.show()







#  --------------------------------------------------------------

# 7.
notes = pd.read_csv("data/notes.txt", sep=r"\s+").to_numpy()
n, p = notes.shape
M = np.eye(p)
Dp = 1/n * np.eye(n)

# 8.

notes_mean = notes.mean(axis=0) # axis=0 pour avoir la moyenne par colonne
notes = notes - notes_mean

# 9.
V = notes.T @ Dp @ notes
val_p, vec_p = linalg.eigh(V @ M)
val_p = val_p[::-1]
vec_p = vec_p[:, ::-1]


# 10.

percents = 100 * val_p / sum(val_p)
# plt.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5"], percents)
# plt.show()

# 11.

U = vec_p
C = notes @ U

# print(C)


# 12.

W = notes @ M @ notes.T

valp, vecp = linalg.eigh(W @ Dp)

