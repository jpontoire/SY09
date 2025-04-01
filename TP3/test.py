import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import add_labels
import numpy.linalg as linalg
from math import sqrt

# 1.
data = pd.read_csv("data/notes.txt", sep=r"\s+")
# plt.scatter(x=data.math, y=data.scie)
# add_labels(data.math, data.scie, data.index)
# plt.show()

# 2.1
data_tmp = data.reset_index() # sert à rajouter un index, sinon ça explose
melt = data_tmp.melt(id_vars=["index"], var_name="matiere", value_name="note")
# sns.barplot(data=melt, x="index", y="note", hue="matiere")
# plt.show()

# 2.2
corr = data.corr()
# sns.heatmap(corr)
# plt.show()
# math et sciences
# français et latin
# arts pas corrélé avec le reste

# matrice bloc-diagonale

# 3.
# ------ sciences
# plt.scatter(data.math, data.scie)
# add_labels(data.math, data.scie, data.index)
# plt.show()
# brig, didi et moni qui se détachent

# ------ littéraire
# plt.scatter(data.fran, data.lati)
# add_labels(data.fran, data.lati, data.index)
# plt.show()
# didi, evel, pier et moni

# ------ arts
# ax = sns.stripplot(data["d-m"])
# loc = ax.get_children()[0].get_offsets().data
# add_labels(*loc.T, data.index)
# plt.show()
# représente sur un axe et décale si il y en a qui ont la même valeur

# 4.
# plt.scatter((data.math + data.scie)/2, (data.fran + data.lati)/2)
# add_labels((data.math + data.scie)/2, (data.fran + data.lati)/2, data.index)
# plt.show()
# on perd l'information de si un élève est bon en maths et mauvais en sciences
# on ne peut pas le différencier d'un élève qui est moyen dans les deux


# 8.
X = data[["math", "scie", "fran", "lati", "d-m"]].to_numpy()
V = np.cov(X, rowvar=False, bias=True)
print(np.diag(V).sum())
# rowvar : individus en ligne et variables en colonnes
# bias : à true elle sera non corrigée

# 9.
A1 = np.array(
    [
        [0.5, 0, 0.5, 0, 0],
        [0.5, 0, -0.5, 0, 0],
        [0, 0.5, 0, 0.5, 0],
        [0, 0.5, 0, -0.5, 0],
        [0, 0, 0, 0, 1],
    ]  
)

V1 = np.cov(X @ linalg.inv(A1).T, rowvar=False, bias=True)
print(np.diag(V1).sum())

# 10.
t = sqrt(2)/2
B1 = np.array(
    [
        [t, 0, t, 0, 0],
        [t, 0, -t, 0, 0],
        [0, t, 0, t, 0],
        [0, t, 0, -t, 0],
        [0, 0, 0, 0, 1],
    ]  
)

V2 = np.cov(X @ B1, rowvar=False, bias=True)
print(np.diag(V2).sum())

B2 = np.array(
    [
        [t, 0, t, 0, 0],
        [0, t, 0, t, 0],
        [t, 0, -t, 0, 0],
        [0, t, 0, -t, 0],
        [0, 0, 0, 0, 1],
    ]  
)

V3 = np.cov(X @ B2, rowvar=False, bias=True)
print(np.diag(V3).sum())