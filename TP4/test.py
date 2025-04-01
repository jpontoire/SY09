import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import add_labels
import scipy.linalg as linalg

# 1.
notes = pd.read_csv("data/notes.txt", sep="\\s+")
cls = PCA(n_components=5)
pcs = cls.fit_transform(notes)

# plt.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5"], cls.explained_variance_ratio_)
# plt.show()
# les axes 4 et 5 ont une inertie quasi nulle, on pourrait les retirer pour compresser les données


# 2.
# plt.scatter(pcs[:, 0], pcs[:, 1])
# add_labels(pcs[:, 0], pcs[:, 1], notes.index)
# plt.show()

# 3.
# plt.scatter(pcs[:, 0], pcs[:, 1])
# add_labels(pcs[:, 0], pcs[:, 1], notes.index)

etu = pd.DataFrame([[8.0, 6.0, 10.0, 9, 14], [10, 11, 4.5, 8.0, 6]], columns=notes.columns, index=["Alice", "Steve"])
pcs_2 = cls.transform(etu)
# plt.scatter(pcs_2[:, 0], pcs_2[:, 1])
# add_labels(pcs_2[:, 0], pcs_2[:, 1], etu.index)
# plt.show()


# 4.
crabs = pd.read_csv("data/crabs.csv", sep=r"\s+")
crabsquant = crabs.iloc[:, 3:8]

# 5.
cbs = crabs.melt(id_vars=["sp", "sex", "index"])
# sns.boxplot(x="variable", y="value", data=cbs, hue="sex")

# sns.scatterplot(x="FL", y="RW", data=crabs, hue="sp", style="sex")

# sns.scatterplot(x="FL", y="CL", data=crabs, hue="sp", style="sex")
# très forte corrélation entre CL et FL / RW et FL

# plt.show()


# 6.
cls = PCA(n_components=5)
pcs_crabs = cls.fit_transform(crabsquant)

df_crabs = pd.DataFrame(pcs_crabs, columns=[f"PC{i}" for i in range(1,6)])
# sns.scatterplot(x="PC1", y="PC2", hue=crabs.sp, style=crabs.sex, data=df_crabs)
# plt.show()

# plt.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5"], cls.explained_variance_ratio_)
# plt.show()

# Axe 1 : taille des crabes
# On peut essayer avec les autres axes

# sns.scatterplot(x="PC2", y="PC3", hue=crabs.sp, style=crabs.sex, data=df_crabs)
# plt.show()
# séparations beaucoup plus visibles avec cette approche


# ratio

# on peut additionner toutes les caractéristiques d'un crabe et considérer que c'est la "taille", et diviser toutes les caractéristiques par cette taille

# on peut chercher la variable qui maximise la somme des corrélations à toutes les variables, et on considère que c'est la taille





# 7.
notes = pd.read_csv("data/notes.txt", sep="\\s+").to_numpy()
n, p = notes.shape
M = np.eye(p)
Dp = 1/n * np.eye(n)


