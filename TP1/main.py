import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1.
X = pd.read_csv("data/sy02-p2016.csv")

# 2.
# print(X.shape)

# 3.
X2 = pd.read_csv("data/sy02-p2016-2.csv", sep="&")
X3 = pd.read_csv("data/sy02-p2016-3.csv", sep="\t")
X4 = pd.read_csv("data/sy02-p2016-4.csv", sep=";")
X5 = pd.read_csv("data/sy02-p2016-5.csv", sep =" ", index_col=0)

# 4.
X.specialite = pd.Categorical(X.specialite)
X.statut = pd.Categorical(X.statut)
X["dernier diplome obtenu"] = pd.Categorical(X["dernier diplome obtenu"])
type_correcteur = pd.CategoricalDtype(
    pd.concat([X["correcteur median"], X["correcteur final"]]).dropna().unique()
)
X["correcteur median"] = X["correcteur median"].astype(type_correcteur)
X["correcteur final"] = X["correcteur final"].astype(type_correcteur)
X.resultat = pd.Categorical(X.resultat, categories=["ABS", "F", "FX", "E", "D", "C", "B", "A"], ordered=True)
# print(X.dtypes)

# 5.
effectifs = pd.read_csv("data/effectifs.csv")
semestres = effectifs.Semestre.str[-5:]
# print(semestres)

# 6.
effectifs = effectifs.assign(Saison=semestres.str[0])
effectifs = effectifs.assign(Année=semestres.str[1:])
effectifs.drop(columns="Semestre", inplace=True)
# print(effectifs)

# 7.
effectifs = effectifs.melt(id_vars=["Saison", "Année"], value_name="Effectif", var_name="UV")
effectifs = effectifs.dropna()
effectifs.Effectif = effectifs.Effectif.astype(int)
# print(effectifs)

# 8.
iris = sns.load_dataset("iris")
iris = iris.melt(id_vars=["species"])
# print(iris)

# 9.
iris = iris.assign(type=iris.variable.str[:5], dimension=iris.variable.str[6:])
iris = iris.drop(columns="variable")
# print(iris)

# 10.
babies = pd.read_csv("data/babies23.data", sep=r"\s+")
babies = babies[["wt", "gestation", "parity", "age", "ht", "wt.1", "smoke", "ed"]]
babies.columns = ["bwt", "gestation", "parity", "age", "height", "weight", "smoke", "education"]
# print(babies)

# 11.
# plt.hist(babies.gestation)
# plt.show()
# valeurs absurdent au niveau de 1000, valeurs inconnues

# 12.
babies.loc[babies.bwt == 999, "bwt"] = np.nan
babies.loc[babies.gestation == 999, "gestation"] = np.nan
babies.loc[babies.age == 99, "age"] = np.nan
babies.loc[babies.height == 99, "height"] = np.nan
babies.loc[babies.weight == 999, "weight"] = np.nan
babies.loc[babies.smoke == 9, "smoke"] = np.nan
babies.loc[babies.education == 9, "education"] = np.nan

# 13.
mask = babies.smoke == 1
babies.smoke = babies.smoke.astype(object)
babies.loc[mask, "smoke"] = "Smoking"
babies.loc[~mask, "smoke"] = "NonSmoking"
babies.smoke = babies.smoke.astype("category")
# print(babies.smoke)

# 14.
def Sijk(d, i, j, k):
    return 2 * d[i, j]**2 * d[i, k]**2 + 2 * d[i, j]**2 * d[j, k]**2 + 2 * d[i, k]**2 * d[j, k]**2 - d[j, k]**4 - d[i, k]**4 - d[i, j]**4

def Smin(d):
    n = d.shape[0]
    return min(
        Sijk(d,i,j,k)
        for i in range(n)
        for j in range(n) if j > i
        for k in range(n) if k > j
    )

# 15.
def add_g(d, g):
    d = d + g
    np.fill_diagonal(d, 0) # c'est une distance / dissimilarité donc la diagonale est nulle
    return d

rng = np.random.default_rng()
N = 5
d = rng.exponential(scale=1, size=(N, N))
d = (d + d.T) / 2
np.fill_diagonal(d, 0)

v_min = - d[d > 0].min()
gammas = np.linspace(v_min, 2, 100)
S_mins = [Smin(add_g(d, c)) for c in gammas]

# plt.plot(gammas, S_mins)
# plt.show()


# 16.
print(max(d[i, j] - d[i, k] - d[j, k]
          for i in range(N)
          for j in range(N)
          for k in range(N)))
# même valeur que celle où le signe change question d'avant

# 17.
# sur feuille

# 18.


# 19.
# A pas proximité car une valeur négative
# B pas similarité ou dissimilarité car diagonale pas nulle ou supérieure au reste
# C dissimilarité car diagonale nulle
# D pas symétrique
# E similarité
# F similarité
# G X
# H valeurs négatives

# 20.
