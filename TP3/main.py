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
data_tmp = data.reset_index() # sert à rajouter un index, sinon ça explose au melt
melt = data_tmp.melt(id_vars=["index"], var_name="matiere", value_name="note")
# sns.barplot(data=melt, x="index", y="note", hue="matiere")
# plt.show()

# 2.2
heat = data.corr()
# sns.heatmap(data=heat)
# plt.show()
# math-sciences
# français-latin

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
# sns.stripplot(data["d-m"])

# plt.show()
# à finir


# 4.
# plt.scatter((data.math + data.scie)/2, (data.fran + data.lati)/2)
# add_labels((data.math + data.scie)/2, (data.fran + data.lati)/2, data.index)
# plt.show()

# 8.
X = data[["math", "scie", "fran", "lati", "d-m"]].to_numpy()
C = np.cov(X, bias=True, rowvar=False)
# print(np.diag(C).sum())

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

A1_inv = linalg.inv(A1)

C0 = np.cov(X @ A1_inv.T, bias=True, rowvar=False)
# print(np.diag(C0).sum())

# 10.
t = sqrt(2)/2
B1 = np.array([
    [t, 0, t, 0, 0],
    [t, 0, -t, 0, 0],
    [0, t, 0, t, 0],
    [0, t, 0, -t, 0],
    [0, 0, 0, 0, 1]
])

C1 = np.cov(X @ B1, bias=True, rowvar=False)
# print(np.diag(C1).sum())

B2 = np.array([
    [t, 0, t, 0, 0],
    [0, t, 0, t, 0],
    [t, 0, -t, 0, 0],
    [0, t, 0, -t, 0],
    [0, 0, 0, 0, 1]
])
C2 = np.cov(X @ B2, bias=True, rowvar=False)
# print(np.diag(C2).sum())


# 11.
# print(np.diag(C))
# print(np.diag(C1))
# print(np.diag(C2))

# on choisit les axes définis par les deux premiers vecteurs de B1 car ce sont ceux qui expliquent la plus grande partie de l'inertie du jeu de données
# en faisant ça on perd l'info du 5ème axe (les arts) qui était mentionné que dans le dernier vecteur

# 12.
B3 = 1/2 * np.array([
    [1,1,1,1,0],
    [1,1,-1,-1,0],
    [1,-1,-1,1,0],
    [1,-1,1,-1,0],
    [0,0,0,0,2]
])

C3 = np.cov(X @ B3, bias=True, rowvar=False)

# print(np.diag(C3)[:2].sum())
# print(np.diag(C1)[:2].sum())

# print(np.diag(C3)[0].sum())
# print(np.diag(C1)[0].sum())

# print(np.diag(C3)[1].sum())
# print(np.diag(C1)[1].sum())

# même inertie totale
# même inertie expliquée par les deux axes
# meilleure inertie expliquée par le 1er axe pour B3


# 13.
rng = np.random.default_rng()

def inerties_cumulées():
    U, _ = linalg.qr(rng.normal(size=(5, 5)))
    C = np.cov(X @ U, bias=True, rowvar=False)
    return np.cumsum(np.diag(C))


# 14.
data_2 = [inerties_cumulées() for i in range(20)]
X0 = pd.DataFrame(data_2, columns=[f"Axe {i+1}" for i in range(5)])
X0.index.name = "Base"
X0 = X0.reset_index()
X0 = X0.melt(id_vars="Base", var_name="Axe", value_name="Inertie Cumulée")
# print(X0)

# 15.
# sns.pointplot(data=X0, x="Axe", y="Inertie Cumulée", hue="Base")
# plt.show()

# la courbe devrait dominer toutes les autres
# il n'y en a pas ici, ce qui est logique puisque les bases sont générées aléatoirement


# 16.
Bx = np.array([[0.515, -0.567, 0.051, -0.289, -0.573],
[0.507, -0.372, 0.014, 0.553, 0.546],
[0.492, 0.65, -0.108, 0.394, -0.41],
[0.485, 0.323, -0.023, -0.674, 0.453],
[0.031, 0.113, 0.992, 0.034, -0.013]])

# ax = sns.pointplot(data=X0, x="Axe", y="Inertie Cumulée", hue="Base")
# ax.get_legend().remove()

C = np.cov(X @ Bx, bias=True, rowvar=False)
csx = np.cumsum(np.diag(C))

# ax.plot(range(5), csx, 'k-o', zorder=100)
# plt.show()

# elle domine toutes les autres


# 17.
