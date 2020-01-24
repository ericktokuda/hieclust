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

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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

def generate_uniform(samplesz, ndims):
    """Generate uniform data

    Args:

    Returns:
    np.ndarray: nxm row
    """
    return np.random.rand(samplesz, ndims)

def generate_multivariate_normal(samplesz, ndims, ncenters, mus=[], cov=[]):
    """Generate multinomial data

    Args:

    Returns:
    np.ndarray: nxm row
    """

    x = np.ndarray((samplesz, ndims), dtype=float)

    truncsz = samplesz // ncenters
    partsz = [truncsz] * ncenters
    diff = samplesz - (truncsz*ncenters)
    partsz[-1] += diff

    if len(mus) == 0:
        mus = np.random.rand(ncenters, ndims)
        cov = np.eye(ndims)

    ind = 0
    for i in range(ncenters):
        mu = mus[i]
        x[ind:ind+partsz[i]] = np.random.multivariate_normal(mu, cov, size=partsz[i])
        ind += partsz[i]
    return x

def generate_exponential(samplesz, ndims, ncenters, mus=[]):
    """Generate multinomial data

    Args:

    Returns:
    np.ndarray: nxm row
    """

    x = np.ndarray((samplesz, ndims), dtype=float)

    truncsz = samplesz // ncenters
    partsz = [truncsz] * ncenters
    diff = samplesz - (truncsz*ncenters)
    partsz[-1] += diff

    if len(mus) == 0:
        mus = np.random.rand(ncenters, ndims)
        cov = np.eye(ndims)

    ind = 0
    for i in range(ncenters):
        mu = mus[i]
        for j in range(ndims):
            x[ind:ind+partsz[i], j] = np.random.exponential(size=partsz[i])
        ind += partsz[i]

    return x

def generate_power(samplesz, ndims, ncenters, power, mus=[]):
    """Generate multinomial data

    Args:

    Returns:
    np.ndarray: nxm row
    """

    x = np.ndarray((samplesz, ndims), dtype=float)

    truncsz = samplesz // ncenters
    partsz = [truncsz] * ncenters
    diff = samplesz - (truncsz*ncenters)
    partsz[-1] += diff

    if len(mus) == 0:
        mus = np.random.rand(ncenters, 2)
        cov = np.eye(2)

    ind = 0
    for i in range(ncenters):
        mu = mus[i]
        xs = 1 - np.random.power(a=power, size=partsz[i])
        ys = 1 - np.random.power(a=power, size=partsz[i])
        x[ind:ind+partsz[i], 0] = xs
        x[ind:ind+partsz[i], 1] = ys
        ind += partsz[i]
    return x

def plot_scatter(x, ax, ndims):
    """Scatter plot

    Args:
    x(np.ndarray): nx2 array, being n the number of points
    """
    if ndims == 2:
        ax.scatter(x[:,0], x[:,1])
    elif ndims == 3:
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])

def generate_data(samplesz, ndims):
    """Synthetic data

    Args:
    n(int): size of each sample

    Returns:
    list of np.ndarray: each element is a nx2 np.ndarray
    """

    data = []

    # 0 cluster
    data.append(generate_uniform(samplesz, ndims))

    # 1 cluster (gaussian)
    mus = np.zeros((1, ndims))
    cov = np.eye(ndims) * 0.15
    data.append(generate_multivariate_normal(samplesz, ndims, ncenters=1,
                                             mus=mus, cov=cov))

    # 1 cluster (power)
    mus = np.zeros((1, ndims))
    data.append(generate_power(samplesz, ndims, ncenters=1, power=3, mus=mus))

    # 1 cluster (exponential)
    mus = np.zeros((1, ndims))
    data.append(generate_exponential(samplesz, ndims, ncenters=1, mus=mus))

    # 2 clusters (gaussians)
    c = 0.7
    mus = np.ones((2, ndims))*c; mus[1, :] *= -1
    cov = np.eye(ndims) * 0.1
    data.append(generate_multivariate_normal(samplesz, ndims, ncenters=2,
                                             mus=mus,cov=cov))

    # 2 clusters (gaussians)
    cov = np.eye(ndims) * 0.01
    data.append(generate_multivariate_normal(samplesz, ndims, ncenters=2,
                                             mus=mus, cov=cov))

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

    ndims = 2
    data = generate_data(samplesz, ndims)
    ndistribs = len(data)

    nrows = ndistribs
    ncols = nlinkagemeths + 1
    fig = plt.figure(figsize=(ncols*10, nrows*10))
    ax = np.array([[None]*ncols]*nrows)

    nsubplots = nrows * ncols

    for subplotidx in range(nsubplots):
        i = subplotidx // ncols
        j = subplotidx % ncols

        if ndims == 3 and j == 0: proj = '3d'
        else: proj = None

        ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1,
                               projection=proj)

    for i in range(ndistribs):
        x = data[i]
        plot_scatter(x, ax[i, 0], ndims)
        for j, l in enumerate(linkagemeths):
            generate_dendrogram(x, l, ax[i, j+1])

    for ax_, col in zip(ax[0, 1:], linkagemeths):
        ax_.set_title(col, size=36)

    plottitles = [ 'Uniform', '1_Normal_0.1', '1_Exponential',
        '1_Power_2', '2_Normal_0.1', '1_Exponential_0.01', ]

    for ax_, row in zip(ax[:, 0], plottitles):
        ax_.set_ylabel(row + '  ', rotation=90, size=36)

    plt.savefig('/tmp/foo.pdf')

if __name__ == "__main__":
    main()

