import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, pi, exp
from scipy.stats import norm
import random

norm.pdf()
# loc = esperance


titanic = sns.load_dataset("titanic")
iris = sns.load_dataset("iris")

# 1.
# sns.boxplot(x=titanic.age)
# plt.show()
# age médian environ à 28 ans

# 2.
# sns.histplot(x=titanic.age)
# plt.show()
# bins = nombre de rectangles
# kde = courbe de densité ?

# 3.
# sns.countplot(x="class", data=titanic)
# plt.show()

# 4.
# sns.scatterplot(x="sepal_length", y="sepal_width", data=iris)
# plt.show()

# 5.
# sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
# plt.show()

# 6.
# sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", size="petal_length", data=iris)
# plt.show()

# 7.
# corr = iris.corr(numeric_only=True)
# sns.heatmap(corr)
# plt.show()
# à analyser

# 8.
# sns.barplot(
#     x="species",
#     y="sepal_length",
#     data=iris
# )
# plt.show()

# 9.
# sns.boxplot(
#     x="species",
#     y="sepal_length",
#     data=iris
# )
# plt.show()

# 10.
# sns.countplot(
#     x="class",
#     hue="embark_town",
#     data=titanic
# )
# plt.show()

# 11.
# sns.countplot(x="class", hue="who", data=titanic)
# plt.show()

# 12.
# sns.displot(x="age", col="class", kind="hist", data=titanic)
# plt.show()

# 13.
# sns.displot(x="age", col="class", row="who", kind="hist", data=titanic)
# plt.show()

# 14.
# sy02 = pd.read_csv("data/sy02-p2019.csv")
# print(sy02["Note médian"].dtype) 
# # type object car ABS dans les valeurs

sy02 = pd.read_csv("data/sy02-p2019.csv", na_values="ABS")
# print(sy02["Note médian"].dtype)

# 15.
# sns.countplot(x="Branche", hue="Semestre", data=sy02)
# plt.show()

sy02["Note ECTS"] = pd.Categorical(sy02["Note ECTS"], ordered=True)
# sns.countplot(x="Branche", hue="Note ECTS", data=sy02)
# plt.show()

# 16.
# sns.histplot(sy02["Note médian"])
# plt.show()

# sns.histplot(sy02["Note final"])
# plt.show()

# sns.scatterplot(x="Note médian", y="Note final", data=sy02)
# plt.show()

# sns.jointplot(x="Note médian", y="Note final", data=sy02)
# plt.show()

# 17.
X = sy02.melt(value_vars=["Note médian", "Note final"], var_name="Type", value_name="Note")
# sns.boxplot(x="Type", y="Note", data=X)
# plt.show()

# 18.
sy02["Same"] = sy02["Correcteur médian"] == sy02["Correcteur final"]
X = sy02.melt(id_vars=["Same"], value_vars=["Correcteur médian", "Correcteur final"], var_name="Examen", value_name="Correcteur")

# sns.countplot(x="Correcteur", hue="Same", data=X)
# plt.show()

# 19.
# sns.countplot(x="Correcteur", hue="Examen", data=X)
# plt.show()
# ça marche mais il faudrait renommer "Correcteur médian" en "Médian" et "Correcteur final" en "final"




# -------------------------------------------------------------------------------------------------------------------------------------------

def densité(x, ech, h):
    
    norm.pdf()

