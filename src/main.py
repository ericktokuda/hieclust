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

def get_element_ids(z, nelements, clustid):
    if clustid < nelements:
        return [clustid]

    zid = int(clustid - nelements)
    leftid = z[zid, 0]
    rightid = z[zid, 1]
    elids1 = get_element_ids(z, nelements, leftid)
    elids2 = get_element_ids(z, nelements, rightid)
    return np.concatenate((elids1, elids2)).astype(int)

def get_last_joinid_given_relative_height(z, n, relheight):
    maxdist = z[-1, 2]

    if maxdist <= relheight:
        return z.shape[0] - 1

    njoins = z.shape[0]

    accdist = 0
    for i in np.arange(njoins-1, -2, -1):
        l = z[i, 2] / maxdist
        if l < relheight: # inds less than or equal i corresponds to the merges
            break
    return i

def get_last_joinid_given_nclusters(z, n, nclusters):
    njoins = z.shape[0]
    return np.min([np.max([njoins - nclusters, -1]), njoins-1])

def get_clustids_from_joinid(joinid, z, n):
    allids = set(range(n+joinid+1))
    mergedids = set(z[:joinid+1, :2].flatten().astype(int))
    clustids = allids.difference(mergedids)
    return clustids

# def count_nelements_from_clustid(clustid):

def get_joinid_from_clustid(clustid, z, n):
    return clustid - n

def get_dist_from_joinid(joinid, z):
    if joinid < 0 or joinid > len(z):
        print('Index joindid ({}) is out of range'.format(joinid))
    return z[joinid, 2]

def get_clusters_limited_by_dist(dist, z, inclusive=True):
    dists = z[:, 2]

    for i in range(len(z)):
        if z[i, 2] > dist: break

    lastjoinid = i - 1
    # get_element_ids(z, nelements, clustid):

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(0)

    thresh = 0.25 # maximum relative distance between elements in the same cluster
    x1 = np.array([ [0, 0], [0, 4], [10, 0], [10, 1]])
    x2 = np.array([ [0, 0], [0, 2], [0, 4], [0, 6],
                  [10, 0], [10, 1], [10, 2]])
    x3 = np.array([ [0, 0], [0, 3], [0, 7], [0, 10],
                  [10, 0], [10, 1], [10, 1], [10, 3]])
    x4 = np.array([ [0, 0], [0, 3], [0, 7], [0, 10], [0, 15],
                  [10, 0], [10, 1], [10, 2], [10, 3]])
    x = x4

    n = x.shape[0]

    linkagemeth = 'single'
    z = linkage(x, linkagemeth)
    # dendrogram(z)
    # plt.show()
    nclusters = n + z.shape[0]
    minclustsize = 2
    minnclusters = 1
    L = z[-1, 2]
    
    # {joinid \in N | -1  <= joinid <= n}
    print(x)
    print(z)

    started = False # Method should stop from 2clusters on
    clustids = set([n*2-2])

    for i in range(1, n+1): # variable for the number of clusters
        # In each loop we expand the children
        lastjoinid = get_last_joinid_given_nclusters(z, n, i)
        clustid = lastjoinid + n
        print(i, lastjoinid, clustid, clustids)
        # input()

        if len(clustids) > 1:  started = True

        if started and len(clustids) <= minnclusters:
            break

        for j in [0, 1]: # left and right children
            els = get_element_ids(z, n, z[lastjoinid, j])
            if len(els) >= minclustsize:
                clustids.add(int(z[lastjoinid, j]))

        clustids.remove(clustid)

    joinid = clustids.pop() - n
    l = z[joinid, 2]
    print(clustid, (L-l)/L)
    return

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

    plt.savefig('/tmp/{}d.pdf'.format(ndims))

if __name__ == "__main__":
    main()

