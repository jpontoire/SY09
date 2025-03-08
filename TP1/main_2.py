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
X5 = pd.read_csv("data/sy02-p2016-5.csv", sep=" ", index_col=0)

# 4.
X.specialite = pd.Categorical(X.specialite)
X.statut = pd.Categorical(X.statut)
X["dernier diplome obtenu"] = pd.Categorical(X["dernier diplome obtenu"])
correcteur = pd.CategoricalDtype(pd.concat([X["correcteur median"], X["correcteur final"]]).dropna().unique())
X["correcteur median"] = X["correcteur median"].astype(correcteur)
X["correcteur final"] = X["correcteur final"].astype(correcteur)
X.resultat = X.resultat.astype(pd.CategoricalDtype(categories=["ABS", "F", "Fx", "E", "D", "C", "B", "A"], ordered=True))

# 5.
file = pd.read_csv("data/effectifs.csv")
semestre = file.Semestre.str[8:]
# print(semestre)

# 6.
file = file.assign(Saison=semestre.str[0])
file = file.assign(Année=semestre.str[1:])
file = file.drop(columns="Semestre")
# print(file)

# 7.
file = file.melt(id_vars=["Saison", "Année"], var_name="UV", value_name="Effectif").dropna()
file.Effectif = file.Effectif.astype(int)
# print(file)

# 8.
iris = sns.load_dataset("iris")
iris = iris.melt(id_vars="species")
# print(iris)

# 9.
iris = iris.assign(type=iris.variable.str[:5], dimension=iris.variable.str[6:])
iris = iris.drop(columns="variable")
# print(iris)

# 10.
babies = pd.read_csv("data/babies23.data", sep=r"\s+")
babies = babies[["wt", "gestation", "parity", "age", "ht", "wt.1", "smoke", "ed"]]
babies.columns = ["bwt", "gestation", "parity", "age", "height", "weight", "smoke", "education"]

# 11.
# plt.hist(babies.gestation)
# plt.show()

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
print(babies.smoke)

# 14.
