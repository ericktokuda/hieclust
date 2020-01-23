#!/usr/bin/env python3
"""Benchmark on hierarchical clustering methods
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import inconsistent

def generate_dendrogram(x, linkagemeth, ax):

    # print(x)
    # print(np.max(x))
    Z2 = linkage(x, linkagemeth)
    dendrogram(
        Z2,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        ax=ax
    )

def generate_uniform(samplesz):
    """Generate uniform data

    Args:

    Returns:
    np.ndarray: nxm row
    """
    return np.random.rand(samplesz, 2)

def generate_multivariate_normal(samplesz, ncenters, mus=[], cov=[]):
    """Generate multinomial data

    Args:

    Returns:
    np.ndarray: nxm row
    """

    x = np.ndarray((samplesz, 2), dtype=float)

    truncsz = samplesz // ncenters
    partsz = [truncsz] * ncenters
    diff = samplesz - (truncsz*ncenters)
    partsz[-1] += diff

    if len(mus) == 0:
        mus = np.random.rand(ncenters, 2)
        cov = np.eye(2)

    ind = 0
    for i in range(ncenters):
        # mu = np.random.rand(2)
        mu = mus[i]
        x[ind:ind+partsz[i]] = np.random.multivariate_normal(mu, cov,
                                                         size=partsz[i])
        ind += partsz[i]
    return x

def plot_scatter(x, ax):
    """Scatter plot

    Args:
    x(np.ndarray): nx2 array, being n the number of points
    """
    ax.scatter(x[:,0], x[:,1])

def generate_data(samplesz):
    """Synthetic data

    Args:
    n(int): size of each sample

    Returns:
    list of np.ndarray: each element is a nx2 np.ndarray
    """

    data = []
    data.append(generate_uniform(samplesz))

    c = 0.7
    mus = np.ones((2, 2))*c; mus[1, :] *= -1

    cov = np.eye(2) * 0.15
    data.append(generate_multivariate_normal(samplesz, ncenters=2, mus=mus, cov=cov))

    cov = np.eye(2) * 0.012
    data.append(generate_multivariate_normal(samplesz, ncenters=2, mus=mus, cov=cov))

    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    np.set_printoptions(precision=5, suppress=True)

    np.random.seed(0)
    samplesz = 200

    linkagemeths = ['single', 'complete', 'average',
                    'centroid', 'median', 'ward']
    nlinkagemeths = len(linkagemeths)

    data = generate_data(samplesz)
    ndistribs = len(data)

    fig, ax = plt.subplots(ndistribs, nlinkagemeths+1,
                           figsize=((nlinkagemeths+1)*10, ndistribs*10))

    for i in range(ndistribs):
        x = data[i]
        plot_scatter(x, ax[i, 0])
        for j, l in enumerate(linkagemeths):
            generate_dendrogram(x, l, ax[i, j+1])

    plt.savefig('/tmp/foo.pdf')

if __name__ == "__main__":
    main()

