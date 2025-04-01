import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from math import sqrt
from sklearn.manifold import MDS
from src.utils import plot_Shepard, plot_dendrogram
from sklearn.cluster import AgglomerativeClustering

def add_labels(x, y, labels, ax=None):
    """Ajoute les étiquettes `labels` aux endroits définis par `x` et `y`."""

    if ax is None:
        ax = plt.gca()
    for x, y, label in zip(x, y, labels):
        ax.annotate(
            label, [x, y], xytext=(10, -5), textcoords="offset points",
        )

    return ax

# 1.
mut = pd.read_csv("data/mutations2.csv", index_col=0)
# print(mut.shape)

# 2.

aftd = MDS(n_components=2, dissimilarity="precomputed")
res = aftd.fit_transform(mut)

# 3.
# plt.scatter(*res.T)
# add_labels(res[:, 0], res[:, 1], mut.index)
# plt.show()

# 4.
"""
aftd = MDS(n_components=2, dissimilarity="precomputed")
res = aftd.fit_transform(mut)
plot_Shepard(aftd)
plt.show()

aftd = MDS(n_components=3, dissimilarity="precomputed")
res = aftd.fit_transform(mut)
plot_Shepard(aftd)
plt.show()

aftd = MDS(n_components=4, dissimilarity="precomputed")
res = aftd.fit_transform(mut)
plot_Shepard(aftd)
plt.show()

aftd = MDS(n_components=5, dissimilarity="precomputed")
res = aftd.fit_transform(mut)
plot_Shepard(aftd)
plt.show()
"""
# points de plus en plus proches de la droite

# 5.
"""
print(aftd.stress_)
diss, dist = plot_Shepard(aftd, plot=False)
stress = np.sum((diss - dist)**2)
print(stress)
"""

# 6.
"""
iris = sns.load_dataset("iris")
cls = AgglomerativeClustering(linkage="ward", distance_threshold=0, n_clusters=None)
iris_2 = iris.drop(columns="species")
cls.fit(iris_2)
plot_dendrogram(cls)
plt.show()
"""

# 7.
"""
cls = AgglomerativeClustering(linkage="complete", distance_threshold=0, n_clusters=None, metric="precomputed")
cls.fit(mut)
plot_dendrogram(cls, labels=mut.index, orientation="left")
plt.show()
"""

# 8.
model = AgglomerativeClustering(linkage="complete", n_clusters=2, metric="precomputed").fit(mut)
aftd = MDS(n_components=2, dissimilarity="precomputed")
res = aftd.fit_transform(mut)

df = pd.DataFrame({"x": res[:, 0], "y": res[:, 1], "étiquette": model.labels_})
sns.scatterplot(x="x", y="y", hue="étiquette", data=df)
add_labels(res[:, 0], res[:, 1], mut.index)
plt.show()