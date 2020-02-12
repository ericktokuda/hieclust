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
import scipy.stats as stats

from mpl_toolkits.mplot3d import Axes3D

##########################################################
def plot_dendrogram(z, linkagemeth, ax, lthresh, clustids):
    """Call fancy scipy.dendogram with @clustids colored and with a line with height
    given by @lthresh

    Args:
    z(np.ndarray): linkage matrix
    linkagemeth(str): one the allowed linkage methods in the scipy.dendogram arguments
    ax(plt.Axis): axis to plot
    lthres(float): complement of the height
    clustids(list): list of cluster idss
    """

    dists = z[:, 2]
    dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
    z[:, 2] = dists
    n = z.shape[0] + 1
    colors = n * (n - 1) * ['k']
    vividcolors = ['b', 'g', 'r', 'c', 'm']

    for clustid in clustids:
        c = vividcolors.pop()
        f, g = get_descendants(z, n, clustid)
        g = np.concatenate((g, [clustid]))
        for ff in g: colors[ff]  = c

    L = z[-1, 2]
    lineh = (L - lthresh) / L

    epsilon = 0.000001
    dendrogram(
        z,
        color_threshold=lthresh+epsilon,
        # truncate_mode='level',
        truncate_mode=None,
        p=3,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=False,
        show_leaf_counts=True,
        ax=ax,
        link_color_func=lambda k: colors[k],
    )
    ax.axhline(y=lineh, linestyle='--')

##########################################################
def generate_uniform(samplesz, ndims):
    return np.random.rand(samplesz, ndims)

##########################################################
def generate_multivariate_normal(samplesz, ndims, ncenters, mus=[], cov=[]):
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

##########################################################
def generate_exponential(samplesz, ndims, ncenters, mus=[]):
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

##########################################################
def generate_power(samplesz, ndims, ncenters, power, mus=[]):
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

##########################################################
def plot_scatter(x, ax, ndims):
    if ndims == 2:
        ax.scatter(x[:,0], x[:,1])
    elif ndims == 3:
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])

##########################################################
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

##########################################################
def get_descendants(z, nleaves, clustid):
    """Get all the descendants from a given cluster id

    Args:
    z(np.ndarray): linkage matrix
    nleaves(int): number of leaves
    clustid(int): cluster id

    Returns:
    np.ndarray, np.ndarray: (leaves, links)
    """

    if clustid < nleaves:
        return [clustid], []

    zid = int(clustid - nleaves)
    leftid = z[zid, 0]
    rightid = z[zid, 1]
    elids1, linkids1 = get_descendants(z, nleaves, leftid)
    elids2, linkids2 = get_descendants(z, nleaves, rightid)
    linkids = np.concatenate((linkids1, linkids2, [leftid, rightid])).astype(int)
    return np.concatenate((elids1, elids2)).astype(int), linkids

##########################################################
def is_child(parent, child, linkageret):
    """Check if @child is a direct child of @parent

    Args:
    parent(int): parent id
    child(int): child id
    linkageret(np.ndarray): linkage matrix

    Returns:
    bool: whether it is child or not
    """

    nleaves = linkageret.shape[0] + 1
    leaves, links = get_descendants(linkageret, nleaves, parent)
    if (child in leaves) or (child in links): return True
    else: return False

##########################################################
def filter_clustering(data, linkageret, minclustsize, minnclusters):
    """Compute relevance according to Luc's method

    Args:
    data(np.ndarray): data with columns as dimensions and rows as points
    linkageret(np.ndarray): linkage matrix

    Returns:
    np.ndarray: array of cluster ids
    float: relevance of this operation
    """

    n = data.shape[0]
    nclusters = n + linkageret.shape[0]
    lastclustid = nclusters - 1
    L = linkageret[-1, 2]

    counts = linkageret[:, 3]

    clustids = []
    for clustcount in range(minclustsize, n):
        if len(clustids) >= minnclusters: break
        joininds = np.where(linkageret[:, 3] == clustcount)[0]

        for joinidx in joininds:
            clid = joinidx + n
            newclust = True
            for other in clustids:
                if is_child(clid, other, linkageret):
                    newclust = False
                    break
            if newclust: clustids.append(clid)

    m = np.max(clustids)
    parent = 2 * n - 1
    for i in range(m + 1, 2 * n - 1):
        allchildrem = True
        for cl in clustids:
            if not is_child(i, cl, linkageret):
                allchildrem = False
                break
        if allchildrem:
            parent = i
            break

    l = linkageret[parent - n, 2]
    acc = 0
    for cl in clustids:
        acc += linkageret[cl - n, 2]

    acc /= len(clustids)
    rel = (L - acc) / L

    return clustids, rel, l / L

##########################################################
def generate_relevance_distrib_all():
    samplesz = 200
    minnclusters = 2
    minrelsize = 0.3
    minclustsize = int(minrelsize * samplesz)
    ndims = 2
    nrealizations = 100

    info('Samplesize:{}, min nclusters:{}, min clustsize:{}'.\
         format(samplesz, minnclusters, minclustsize))

    # linkagemeths = ['single', 'complete', 'average',
                    # 'centroid', 'median', 'ward']
    linkagemeth = 'ward'
    nlinkagemeths = 1
    info('Computing:{}'.format(linkagemeth))

    ndistribs = 4
    # data = generate_data(samplesz, ndims)[:ndistribs]

    nrows = ndistribs
    # ncols = nlinkagemeths + 1
    ncols = 2
    # fig = plt.figure(figsize=(ncols*10, nrows*10))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*10))

    # ax = np.array([[None]*ncols]*nrows)
    fig.suptitle('Sample size:{}, minnclusters:{}, min clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize),
                 fontsize=60)

    rels = []
    for i in range(ndistribs):
        rels.append({'1': [], '2': []})

    for k in range(nrealizations):
        data = generate_data(samplesz, ndims)[:ndistribs]
        for i in range(ndistribs):
            x = data[i]
            z = linkage(x, linkagemeth)
            clustids, rel, dist = filter_clustering(x, z, minclustsize, minnclusters)
            nclusters = len(clustids)
            rels[i][str(nclusters)].append(rel)

    nbins = 10
    for i in range(ndistribs):
        n, xx, _ = ax[i, 0].hist(rels[i]['1'], nbins)
        ax[i, 1].hist(rels[i]['2'])

        ax[i, 0].set_xlim(0, 1)
        plt.text(0.7, 0.9, 'count:{}'.format(len(rels[i]['1'])),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=50, transform = ax[i, 0].transAxes)

        ax[i, 1].set_xlim(0, 1)
        plt.text(0.7, 0.9, 'count:{}'.format(len(rels[i]['2'])),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=50, transform = ax[i, 1].transAxes)

    plottitles = [ 'Uniform', '1_Normal_0.1', '1_Exponential',
                  '1_Power_2', '2_Normal_0.1', '1_Exponential_0.01', ]

    for ax_, row in zip(ax[:, 0], plottitles[:ndistribs]):
        ax_.set_ylabel(row + '  ', rotation=90, size=36)

    plt.savefig('/tmp/rel_distribs.pdf')

##########################################################
def generate_dendograms_all():
    # thresh = 0.25 # maximum relative distance between elements in the same cluster
    # x1 = np.array([ [0, 0], [0, 4], [10, 0], [10, 1]])
    # x2 = np.array([ [0, 0], [0, 2], [0, 4], [0, 6],
    # [10, 0], [10, 1], [10, 2]])
    # x3 = np.array([ [0, 0], [0, 3], [0, 7], [0, 10],
    # [10, 0], [10, 1], [10, 1], [10, 3]])
    x4 = np.array([ [0, 0], [0, 3], [0, 7], [0, 10], [0, 15],
                   [10, 0], [10, 1], [10, 2], [10, 3]])
    # x5 = np.array([[0.54881,0.71519],
    # [0.60276, 0.54488],
    # [0.42365, 0.64589],
    # [0.43759, 0.89177],
    # [0.96366, 0.38344],
    # [0.79173, 0.52889]])

    # x6 = np.random.rand(10, 2)
    # x = x6
    # print(x)
    # plt.scatter(x4[:, 0], x4[:, 1])
    # plt.show()
    minclustsize = 3
    minnclusters = 2
    # z = linkage(x, 'single')
    # print(z)
    # clustids, rel = filter_clustering(x, z, minclustsize, minnclusters)
    # print(clustids, rel)
    # return
    # fig, ax = plt.subplots()
    # ax.clear()
    # plot_dendrogram(x, 'single', ax)
    # plt.show()
    # ax.clear()
    # print(z)
    # rel, ll = compute_relevance(x, z, 2)
    # clustids = get_clusters(z, ll, 2)
    # print(clustids)
    # get_descendants
    # n = x.shape[0]
    # for clustid in clustids:
    # els = get_descendants(z, n, clustid)
    # input(els)
    # plot_dendrogram(x, 'single', ax, ll)
    # plt.show()
    # return

    samplesz = 200
    minnclusters = 2
    minrelsize = 0.3
    minclustsize = int(minrelsize * samplesz)
    ndims = 2

    info('Samplesize:{}, min nclusters:{}, min clustsize:{}'.\
         format(samplesz, minnclusters, minclustsize))

    linkagemeths = ['single', 'complete', 'average',
                    'centroid', 'median', 'ward']
    nlinkagemeths = len(linkagemeths)
    info('Computing:{}'.format(linkagemeths))

    data = generate_data(samplesz, ndims)
    ndistribs = len(data)

    nrows = ndistribs
    ncols = nlinkagemeths + 1
    fig = plt.figure(figsize=(ncols*10, nrows*10))
    ax = np.array([[None]*ncols]*nrows)
    fig.suptitle('Sample size:{}, minnclusters:{}, min clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize),
                 fontsize=60)

    nsubplots = nrows * ncols

    for subplotidx in range(nsubplots):
        i = subplotidx // ncols
        j = subplotidx % ncols

        if ndims == 3 and j == 0: proj = '3d'
        else: proj = None

        ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1, projection=proj)

    for i in range(ndistribs):
        x = data[i]
        plot_scatter(x, ax[i, 0], ndims)

        for j, l in enumerate(linkagemeths):
            z = linkage(x, l)
            clustids, rel, dist = filter_clustering(x, z, minclustsize, minnclusters)
            ll = rel
            plot_dendrogram(z, l, ax[i, j+1], ll, clustids)
            plt.text(0.7, 0.9, '{}, rel:{:.3f}'.format(len(clustids), rel),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=50, transform = ax[i, j+1].transAxes)

    for ax_, col in zip(ax[0, 1:], linkagemeths):
        ax_.set_title(col, size=36)

    plottitles = [ 'Uniform', '1_Normal_0.1', '1_Exponential',
                  '1_Power_2', '2_Normal_0.1', '1_Exponential_0.01', ]

    for ax_, row in zip(ax[:, 0], plottitles):
        ax_.set_ylabel(row + '  ', rotation=90, size=36)

    plt.savefig('/tmp/{}d.pdf'.format(ndims))

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(0)
    # generate_dendograms_all()
    generate_relevance_distrib_all()

##########################################################
if __name__ == "__main__":
    main()

