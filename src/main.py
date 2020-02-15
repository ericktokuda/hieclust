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

    epsilon = 0.0000
    dendrogram(
        z,
        color_threshold=lthresh+epsilon,
        # truncate_mode='level',
        truncate_mode=None,
        # p=10,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=False,
        show_leaf_counts=True,
        ax=ax,
        link_color_func=lambda k: colors[k],
    )
    ax.axhline(y=lineh, linestyle='--')
    ax.axhline(y=1, linestyle='--')

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
        xs = 1 - np.random.power(a=power+1, size=partsz[i])
        ys = 1 - np.random.power(a=power+1, size=partsz[i])
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

    data = {}

    # 0 cluster
    # data.append(generate_uniform(samplesz, ndims))
    data['1,uniform'] = generate_uniform(samplesz, ndims)

    # 1 cluster (gaussian)
    mus = np.zeros((1, ndims))
    cov = np.eye(ndims) * 0.15
    data['1,gaussian'] = generate_multivariate_normal(samplesz, ndims, ncenters=1,
                                             mus=mus, cov=cov)
    # 1 cluster (linear)
    mus = np.zeros((1, ndims))
    data['1,linear'] = generate_power(samplesz, ndims, ncenters=1, power=1, mus=mus)

    # 1 cluster (power)
    mus = np.zeros((1, ndims))
    data['1,power'] = generate_power(samplesz, ndims, ncenters=1, power=2, mus=mus)

    # 1 cluster (exponential)
    mus = np.zeros((1, ndims))
    data['1,exponential'] = generate_exponential(samplesz, ndims, ncenters=1, mus=mus)

    # 2 clusters (gaussians)
    c = 0.7
    mus = np.ones((2, ndims))*c; mus[1, :] *= -1
    cov = np.eye(ndims) * 0.1
    data['2,gaussian,std0.1'] = generate_multivariate_normal(samplesz, ndims,
                                                              ncenters=2,
                                                              mus=mus,cov=cov)

    # 2 clusters (gaussians)
    cov = np.eye(ndims) * 0.01
    data['2,gaussian,std0.01'] = generate_multivariate_normal(samplesz, ndims,
                                                               ncenters=2,
                                                               mus=mus,cov=cov)

    return data, len(data.keys())

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
    for clustcount in range(minclustsize, n): # Find the clustids
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

    if len(clustids) == 1:
        l = linkageret[clustids[0] - n, 2]
        rel = (L - l) / L
        return clustids, rel
        

    m = np.max(clustids)
    parent = 2 * n - 1
    for i in range(m + 1, 2 * n - 1): # Find the parent id
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

    clustids = sorted(clustids)[:2]
    return clustids, rel

##########################################################
def compute_gtruth_vectors(data, nrealizations):
    """Compute the ground-truth given by Luc method

    Args:
    data(dict): dict with key 'numclust,method,param' and list as values

    Returns:
    dict: key 'numclust,method,param' and list as values
    """
    gtruths = {}
    for i, k in enumerate(data):
        nclusters = int(k.split(',')[0])
        gtruths[k] = np.zeros(2)
        gtruths[k][nclusters-1] = nrealizations

    return gtruths

##########################################################
def generate_relevance_distrib_all():
    samplesz = 200
    minnclusters = 2
    minrelsize = 0.3
    minclustsize = int(minrelsize * samplesz)
    ndims = 10
    nrealizations = 100

    info('Nrealizations:{}, Samplesize:{}, min nclusters:{}, min clustsize:{}'.\
         format(nrealizations, samplesz, minnclusters, minclustsize))

    linkagemeth = 'single' # 'complete', 'average', 'centroid', 'median', 'ward'
    nlinkagemeths = 1
    info('Computing:{}'.format(linkagemeth))

    data, ndistribs = generate_data(samplesz, ndims)
    gtruths = compute_gtruth_vectors(data, nrealizations)
    nrows = ndistribs
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))

    fig.suptitle('Sample size:{}, minnclusters:{}, min clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize),
                 fontsize=60)

    rels = dict((el, [[], []]) for el in data.keys())

    for _ in range(nrealizations): # Compute relevances
        data, _ = generate_data(samplesz, ndims)

        for i, distrib in enumerate(data):
            z = linkage(data[distrib], linkagemeth)
            clustids, rel = filter_clustering(data[distrib], z, minclustsize,
                                                    minnclusters)
            rels[distrib][len(clustids)-1].append(rel)

    # Compute the summarized vector
    v = dict((el, np.zeros(2)) for el in data.keys())
    for i, distrib in enumerate(data):
        for j, rel in enumerate(rels[distrib]):
            v[distrib][j] = np.sum(rel)

    # Compute the difference vector
    diff = dict((el, np.zeros(2)) for el in data.keys())
    diffnorms = {}
    for i, distrib in enumerate(data):
        # print((gtruths[distrib]), (v[distrib]))
        diff[distrib] = gtruths[distrib] - v[distrib]
        diffnorms[distrib] = np.linalg.norm(diff[distrib])
    
    nbins = 10
    for i, distrib in enumerate(data): # Plot
        for j in range(2): # Plot
            ax[i, j].hist(rels[distrib][j], nbins)
            ax[i, j].set_xlim(0, 1)
            plt.text(0.5, 0.9, '{} cluster, n:{}'.\
                     format(j+1, len(rels[distrib][j])),
                     ha='center', va='center',
                     fontsize=40, transform = ax[i, j].transAxes)
        origin = np.zeros(2)
        colours = ['r', 'g', 'b']
        xs = np.array([gtruths[distrib][0], v[distrib][0]])
        ys = np.array([gtruths[distrib][1], v[distrib][1]])
        ax[i, 2].quiver(origin, origin, xs, ys, color=colours, width=.03,
                        scale=nrealizations, headwidth=1, headlenght=3,
                        alpha=0.5)

        ax[i, 2].set_xlim(0, nrealizations)
        ax[i, 2].set_ylim(0, nrealizations)

        plt.text(0.5, 0.9, 'moddiff:{:.2f}'.format(diffnorms[distrib]),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=40, transform = ax[i, 2].transAxes)

        ax[i, 2].set_ylabel('2 clusters', fontsize='xx-large')
        ax[i, 2].set_xlabel('1 cluster', fontsize='xx-large')

    for i, distrib in enumerate(data): # Plot
        ax[i, 0].set_ylabel('{}'.format(distrib), rotation=90, size=36)

    plt.savefig('/tmp/rel_distribs.pdf')

##########################################################
def generate_dendograms_all():
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

    data, ndistribs = generate_data(samplesz, ndims)

    nrows = ndistribs
    ncols = nlinkagemeths + 1
    fig = plt.figure(figsize=(ncols*10, nrows*10))
    ax = np.array([[None]*ncols]*nrows)

    fig.suptitle('Sample size:{}, minnclusters:{}, min clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize), fontsize=60)

    nsubplots = nrows * ncols

    for subplotidx in range(nsubplots):
        i = subplotidx // ncols
        j = subplotidx % ncols

        if ndims == 3 and j == 0: proj = '3d'
        else: proj = None

        ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1, projection=proj)

    for i, k in enumerate(data):
        nclusters = int(k.split(',')[0])
        plot_scatter(data[k], ax[i, 0], ndims)

        for j, l in enumerate(linkagemeths):
            z = linkage(data[k], l)
            clustids, rel = filter_clustering(data[k], z, minclustsize,
                                                    minnclusters)
            plot_dendrogram(z, l, ax[i, j+1], rel, clustids)
            plt.text(0.7, 0.9, '{}, rel:{:.3f}'.format(len(clustids), rel),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=50, transform = ax[i, j+1].transAxes)

    for ax_, col in zip(ax[0, 1:], linkagemeths):
        ax_.set_title(col, size=36)

    for i, k in enumerate(data):
        ax[i, 0].set_ylabel(k, rotation=90, size=36)

    plt.savefig('/tmp/{}d.pdf'.format(ndims))

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.INFO)

    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(0)
    # generate_dendograms_all()
    generate_relevance_distrib_all()

##########################################################
if __name__ == "__main__":
    main()

