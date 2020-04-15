from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

np.random.seed(0)
eps = .05
x = np.array([
    [1,1],
    [2,1],
    [4,1],
    [1,10],
    [1.5,10],
    [4,10],
    ])
Z = linkage(x, 'ward')
print(Z)

fig, ax = plt.subplots(2, figsize=(25, 10))
ax[0].scatter(x[:, 0], x[:, 1])
for i, coords in enumerate(x):
    ax[0].annotate(str(i), (coords[0]+eps, coords[1]+eps))
dendrogram(
    Z,
    leaf_rotation=90.,
    leaf_font_size=8.,
    ax=ax[1]
)
plt.show()
