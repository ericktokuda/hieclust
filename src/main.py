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

from mpl_toolkits.mplot3d import Axes3D

##########################################################
def generate_dendrogram(x, linkagemeth, ax, lthresh=None, clustids=[]):
    z = linkage(x, linkagemeth)
    dists = z[:, 2]
    dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
    z[:, 2] = dists
    n = x.shape[0]
    colors = n * (n - 1) * ['k']
    vividcolors = ['b', 'g', 'r', 'c', 'm']

    # for clustid in clustids:
        # colors[clustid]  = vividcolors.pop()

    # print('##########################################################')
    for clustid in clustids:
        c = vividcolors.pop()
        f = get_element_ids(z, n, clustid)
        # f.append(clustid)
        # print(type(f))
        f = np.append(f, clustid)
        # print((f.shape))
        for ff in f:
            # print(ff, c)
            colors[ff]  = c

    if lthresh:
        epsilon = 0.000001
        # dendrogram(
        # print(linkagemeth, lthresh)
        fancy_dendrogram(
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
            annotate_above=100
        )
    return z

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

    # return data # TODO: remove it

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
def get_element_ids(z, nelements, clustid):
    if clustid < nelements:
        return [clustid]

    zid = int(clustid - nelements)
    leftid = z[zid, 0]
    rightid = z[zid, 1]
    elids1 = get_element_ids(z, nelements, leftid)
    elids2 = get_element_ids(z, nelements, rightid)
    return np.concatenate((elids1, elids2)).astype(int)

##########################################################
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

##########################################################
def get_clustids_from_joinid(joinid, z, n):
    allids = set(range(n+joinid+1))
    mergedids = set(z[:joinid+1, :2].flatten().astype(int))
    clustids = allids.difference(mergedids)
    return clustids

##########################################################
def count_nelements_from_clustid(clustid, z):
    npoints = z.shape[0] + 1
    if clustid < npoints:
        return 1
    else:
        joinid = int(clustid - npoints)
        return z[joinid, 3]

##########################################################
def get_joinid_from_clustid(clustid, z, n):
    return clustid - n

##########################################################
def get_dist_from_joinid(joinid, z):
    if joinid < 0 or joinid > len(z):
        print('Index joindid ({}) is out of range'.format(joinid))
    return z[joinid, 2]

##########################################################
def get_clusters_limited_by_dist(dist, z, inclusive=True):
    dists = z[:, 2]

    for i in range(len(z)):
        if z[i, 2] > dist: break

    lastjoinid = i - 1
    # get_element_ids(z, nelements, clustid):

##########################################################
def compute_relevance(data, linkageret, minclustsize):
    """Compute relevance according to Luc's method

    Args:
    data(np.ndarray): data with columns as dimensions and rows as points
    linkageret(np.ndarray): return of the scipy.linkage call
    """

    n = data.shape[0]
    nclusters = n + linkageret.shape[0]
    lastclustid = nclusters - 1
    # minclustsize = int(n * minclustrelsize)
    # minclustsize = 2
    # print(minclustsize)
    # print(n, minclustrelsize, minclustsize)
    # input()
    minnclusters = 1
    L = linkageret[-1, 2]
    
    started = False # Method should execute from 2clusters on
    clustids = set([lastclustid])
    prevclustid = lastclustid

    nlargegroups = 1
    for depth in range(0, n): # Depth of a node of the tree
        # In each loop we expand the children
        joinid = ( n - 2 ) - depth
        clustid = joinid + n
        if clustid not in clustids: continue

        # print('depth:{}, joinid:{}, clustid:{}, clustids:{}, nlargegroups:{}'.\
              # format(depth, joinid, clustid, clustids, nlargegroups))
        if depth > 0 and not started and nlargegroups > 2:
            # print('started')
            started = True

        if (started and (len(clustids) <= minnclusters)):
            # print('leaving for loop, depth:', depth)
            break

        for j in [0, 1]: # left and right children
            childid = linkageret[joinid, j]
            count = count_nelements_from_clustid(childid, linkageret)
            if count >= minclustsize:
                clustids.add(int(linkageret[joinid, j]))
                nlargegroups += 1

        prevclustid = clustid
        clustids.remove(clustid)
        nlargegroups -= 1
        if (len(clustids) == 0): break

    # clustids.add(prevclustid)
    joinid = ( n - 2 ) - depth + 1 # One above
    l = linkageret[joinid, 2]
    rel = (L-l)/L
    # print(rel, l, clustids)
    # input()
    return rel, l

##########################################################
def get_clusters(linkageret, height, minclustsize):
    """Compute relevance according to Luc's method

    Args:
    data(np.ndarray): data with columns as dimensions and rows as points
    linkageret(np.ndarray): return of the scipy.linkage call
    """
    nleaves = linkageret.shape[0] + 1
    dists = linkageret[:, 2]
    k = np.where(dists == height)[0][-1]
    clustids = set()
    childrem = set()
    # if k == 0: return [nleaves]

    # print(linkageret, minclustsize, k, height)
    for joinid in range(k, 0, -1):
        clustid = joinid + nleaves
        if clustid in childrem: continue
        left = linkageret[joinid, 0]
        right = linkageret[joinid, 1]
        ischild = (left in childrem) or (right in childrem)
        clustsz = linkageret[joinid, 3]
        # print(joinid, left, right, ischild, clustsz)
        # print(minclustsize)
        if (not ischild) and (clustsz >= minclustsize):
            clustids.add(joinid + nleaves)
            childrem.add(left)
            childrem.add(right)

    return list(clustids)

##########################################################
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)
    ax = kwargs.pop('ax', 0)

    if not kwargs.get('no_plot', False):
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                # plt.plot(x, y, 'o', c=c)
                ax.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

##########################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(0)

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
    # x = x4
    # print(x)
    # plt.scatter(x4[:, 0], x4[:, 1])
    # plt.show()
    # fig, ax = plt.subplots()
    # ax.clear()
    # z = generate_dendrogram(x, 'single', ax)
    # plt.show()
    # ax.clear()
    # print(z)
    # rel, ll = compute_relevance(x, z, 2)
    # clustids = get_clusters(z, ll, 2)
    # print(clustids)
    # get_element_ids
    # n = x.shape[0]
    # for clustid in clustids:
        # els = get_element_ids(z, n, clustid)
        # input(els)
    # z = generate_dendrogram(x, 'single', ax, ll)
    # plt.show()
    # return


    samplesz = 200
    minrelsize = 0.45
    minclustsize = int(minrelsize * samplesz)
    info('samplesize:{}, min:{}'.format(samplesz, minclustsize))
    ndims = 2
    linkagemeths = ['single', 'complete', 'average',
                    'centroid', 'median', 'ward']
    # linkagemeths = ['complete']
    nlinkagemeths = len(linkagemeths)

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
            z = generate_dendrogram(x, l, None)
            # print(minclustsize)
            rel, ll = compute_relevance(x, z, minclustsize)
            clustids = get_clusters(z, ll, minclustsize)
            # print(clustids)
            # ax[i, j+1].clear()
            z = generate_dendrogram(x, l, ax[i, j+1], ll, clustids)
            plt.text(0.7, 0.9, '{}, rel:{:.3f}'.format(len(clustids), rel),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=50,
                     transform = ax[i, j+1].transAxes)

    for ax_, col in zip(ax[0, 1:], linkagemeths):
        ax_.set_title(col, size=36)

    plottitles = [ 'Uniform', '1_Normal_0.1', '1_Exponential',
        '1_Power_2', '2_Normal_0.1', '1_Exponential_0.01', ]

    for ax_, row in zip(ax[:, 0], plottitles):
        ax_.set_ylabel(row + '  ', rotation=90, size=36)

    plt.savefig('/tmp/{}d.pdf'.format(ndims))

if __name__ == "__main__":
    main()

