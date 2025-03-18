import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import add_labels


# 1.
data = pd.read_csv("data/notes.txt", sep=r"\s+")
# plt.scatter(x=data.math, y=data.scie)
# add_labels(data.math, data.scie, data.index)
# plt.show()

# 2.1
data_tmp = data.reset_index()
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
plt.scatter((data.math + data.scie)/2, (data.fran + data.lati)/2)
add_labels((data.math + data.scie)/2, (data.fran + data.lati)/2, data.index)
plt.show()