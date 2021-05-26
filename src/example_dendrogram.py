from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

plt.style.use('ggplot')
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

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].scatter(x[:, 0], x[:, 1])
for i, coords in enumerate(x):
    ax[0].annotate(str(i), (coords[0]+eps, coords[1]+eps))

ax[0].set_xlim(0.5, 4.5)
ax[0].set_ylim(0, 12)

dendrogram(
    Z,
    leaf_rotation=90.,
    leaf_font_size=8.,
    ax=ax[1]
)
ax[1].set_ylim(0, 17)
plt.tight_layout(pad=1, w_pad=2)
plt.savefig('/tmp/out.png')
plt.show()
