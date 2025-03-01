import numpy as np
import matplotlib.pyplot as plt 
from math import sqrt

# 1.
t1 = np.logspace(0, 10, 11, base=10)
t2 = np.arange(0, 100, 2)
t3 = np.arange(0, -10, -1)

# print(t1)
# print(t2)
# print(t3)

# 2.
rng = np.random.default_rng()
sample = rng.exponential(scale=0.3, size=1000)
plt.hist(sample)
# plt.show()

# 3.
a = np.array([1, 2, 3])
mean = a.sum()/len(a)
std = sqrt(sum((x - mean)**2 for x in a)/len(a))
# print(mean, np.mean(a))
# print(std, np.std(a))

# 4.
# donne le nombre d'élé de la première dim, équivalent de a.shape[0]

#  5.
a1 = np.array([[1], [-1], [0]])
a2 = np.array([-3, -2, 1])
a3 = np.full((3, 3), -2)
np.fill_diagonal(a3, 0)

# print(a1)
# print(a2)
# print(a3)

# 6.
A1 = (1 + np.arange(8)).reshape(2, -1)
A2 = (1 + np.arange(8)).reshape((2, -1), order="F")

# 7.
def col_to_ligne(mat):
    return np.squeeze(mat)

test = np.array([[1], [-1], [0]])
# print(test)
# print(col_to_ligne(test))

# 8.
def concat_8(a1, a2):
    tmp1 = np.concatenate((a1, a2))
    tmp2 = np.concatenate((-a2, a1))
    return np.concatenate((tmp1, tmp2), axis=1)

a1 = np.array([[1, 2], [2, 1]])
a2 = np.array([[3, 4], [4, 3]])
# print(concat_8(a1, a2))

# 9.
def concat_9(mat, vec, scal):
    tmp1 = np.concatenate((mat, vec[:, np.newaxis]), axis=1)
    tmp2 = np.concatenate((vec, [scal]))
    return np.concatenate((tmp1, tmp2[np.newaxis, :]))

# print(concat_9(np.diag([1, 2]), np.array([3, 4]), 5))

# 10.
def mat_circulante(ligne):
    n = len(ligne)
    c = np.array(ligne.tolist() * (n-1))
    c = c.reshape((n, n-1))
    c = np.concatenate((c, ligne[::-1, np.newaxis]), axis=1)
    return c

ui = np.arange(6)
# print(mat_circulante(ui))

# 11.
# def matrix(n, p):
