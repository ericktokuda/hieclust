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

def generate_multivariate_normal(samplesz, ncenters):
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

    for i in range(ncenters):
        mu = np.random.rand(2)
        cov = np.eye(2) * 0.005
        x[i:i+partsz[i]] = np.random.multivariate_normal(mu, cov,
                                                         size=[partsz[i]])
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
    data.append(generate_multivariate_normal(samplesz, 1)) # k = 0
    data.append(generate_multivariate_normal(samplesz, 3)) # k = 0
    data.append(generate_multivariate_normal(samplesz, 5)) # k = 1
    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('--outdir', required=True, help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(0)  # for repeatability of this tutorial

    # plt.figure(figsize=(10,10))
    linkagemeths = ['single', 'complete', 'average',
                    'centroid', 'median', 'ward']
    n = len(linkagemeths)
    k = 3 # number of data distributions
    samplesz = 100

    fig, ax = plt.subplots(k, n+1, figsize=((n+1)*10, k*10))

    data = generate_data(samplesz)
    
    for i in range(k):
        x = data[i]
        plot_scatter(x, ax[i, 0])
        for j, l in enumerate(linkagemeths):
            generate_dendrogram(x, l, ax[i, j+1])
    plt.savefig('/tmp/foo.pdf')

if __name__ == "__main__":
    main()

