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
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

##########################################################
def multivariate_normal(x, mean, cov):
    """P.d.f. of the multivariate normal when the covariance matrix is positive definite.
    Source: wikipedia"""
    ndims = len(mean)
    B = x - mean
    # print(cov, B)
    # print(np.linalg.solve(cov, B))
    # print(np.sqrt((2 * np.pi)**ndims * np.linalg.det(cov)))
    # print(np.exp(-0.5*(np.linalg.solve(cov, B).T.dot(B))))
    return (1. / (np.sqrt((2 * np.pi)**ndims * np.linalg.det(cov))) *
            np.exp(-0.5*(np.linalg.solve(cov, B).T.dot(B))))

##########################################################
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    z = args[0]
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    inc = inconsistent(z)
    ddata = dendrogram(z, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')

        j = 0
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("{:.3g}".format(inc[j, -1]), (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center', size=7)
            j += 1

        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
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
    return colors[:n]

##########################################################
# def generate_uniform(samplesz, ndims):
    # xlist = np.linspace(0, 1, samplesz)
    # ylist = np.linspace(0, 1, samplesz)
    # # pdf = np.ones((samplesz, samplesz, 3))
    # # pdf[:,:,0], pdf[:,:,1] = np.meshgrid(xlist, ylist)
    # # pdf[:, :2] = np.meshgrid(xlist, ylist)
    # return np.random.rand(samplesz, ndims), pdf

##########################################################
def generate_uniform(samplesz, ndims, cs, rs):
    nrows = samplesz
    data = np.zeros((nrows, ndims))
    pdf = np.zeros((nrows, ndims, 3))
    partsz = get_partition_sizes(samplesz, rs.shape[0])

    dataind = 0
    for i in range(len(rs)):
        c = cs[i, :]
        r = rs[i]
        min_ = c - r
        range_ = 2 * r
        aux = np.random.rand(2*partsz[i], ndims) * range_ + min_
        dists = cdist(aux, np.array([c]))
        inds = np.where(dists <= r)[0][:partsz[i]]
        data[dataind:dataind+partsz[i]] = aux[inds, :]
        dataind += partsz[i]

    return data

##########################################################
def get_partition_sizes(samplesz, npartitions):
    truncsz = samplesz // npartitions
    partsz = [truncsz] * npartitions
    diff = samplesz - (truncsz*npartitions)
    partsz[-1] += diff
    return np.array(partsz)

##########################################################
def generate_multivariate_normal(samplesz, ndims, ncenters, mus=[], covs=[]):
    x = np.ndarray((samplesz, ndims), dtype=float)
    partsz = get_partition_sizes(samplesz, ncenters)

    if len(mus) == 0:
        mus = np.random.rand(ncenters, ndims)
        covs = np.array([np.eye(ndims)] * ncenters)

    # print(mus.shape)
    # print(covs.shape)
    # input()
    ind = 0
    for i in range(ncenters):
        mu = mus[i, :]
        cov = covs[i, :, :]
        # print(mus, mu)
        # print(cov)
        x[ind:ind+partsz[i]] = np.random.multivariate_normal(mu, cov, size=partsz[i])
        ind += partsz[i]
    return x

##########################################################
def generate_exponential(samplesz, ndims, ncenters, mus=[]):
    x = np.ndarray((samplesz, ndims), dtype=float)

    partsz = get_partition_sizes(samplesz, ncenters)

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
def generate_power(samplesz, ndims, power, mus=[]):
    x = np.ndarray((samplesz, ndims), dtype=float)

    ncenters = len(mus)
    if len(mus) == 0:
        mus = np.random.rand(1, 2)
        cov = np.eye(2)
        ncenters = 1

    partsz = get_partition_sizes(samplesz, ncenters)

    ind = 0
    for i in range(ncenters):
        mu = mus[i]
        xs = 1 - np.random.power(a=power+1, size=partsz[i]*2)
        ys = 1 - np.random.power(a=power+1, size=partsz[i]*2)
        xs[np.random.rand(partsz[i]*2) > .5] *= -1
        ys[np.random.rand(partsz[i]*2) > .5] *= -1
        x[ind:ind+partsz[i], 0] = xs[:partsz[i]] + mu[0]
        x[ind:ind+partsz[i], 1] = ys[:partsz[i]] + mu[1]
        ind += partsz[i]
    return x

##########################################################
def plot_scatter(x, ax, ndims, coloursarg=None):
    if ndims == 2:
        ax.scatter(x[:,0], x[:,1], c=coloursarg)
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

    # fig, ax = plt.subplots(2)
    # mus = np.zeros((1, ndims))
    # ax[0] = plot_contour_power(samplesz, ndims, 1, mus, ax[0])
    # ax[1] = plot_contour_power(samplesz, ndims, 5, mus, ax[1])
    # plt.show()

    mu = np.array([[0, 0]])
    r = np.array([.9])
    data['1,uniform,rad0.9'] = generate_uniform(samplesz, ndims, mu, r)
 
    # 1 cluster (gaussian)
    mus = np.zeros((1, ndims))
    covs = np.array([np.eye(ndims) * 0.15])
    data['1,gaussian'] = generate_multivariate_normal(samplesz, ndims, ncenters=1,
                                                      mus=mus, covs=covs)
    # 1 cluster (linear)
    mus = np.zeros((1, ndims))
    data['1,linear'] = generate_power(samplesz, ndims, power=1, mus=mus)

    # 1 cluster (power)
    mus = np.zeros((1, ndims))
    data['1,power'] = generate_power(samplesz, ndims, power=2, mus=mus)

    # 1 cluster (exponential)
    mus = np.zeros((1, ndims))
    data['1,exponential'] = generate_exponential(samplesz, ndims, ncenters=1, mus=mus)

    # 2 clusters (uniform)
    c = 0.7
    mus = np.ones((2, ndims))*c; mus[1, :] *= -1
    rs = np.ones(2) * .9
    data['2,uniform,rad0.9'] = generate_uniform(samplesz, ndims, mus, rs)

    # 2 clusters (uniform)
    rs = np.ones(2) * .5
    data['2,uniform,rad0.5'] = generate_uniform(samplesz, ndims, mus, rs)


    # 2 clusters (gaussians)
    covs = np.array([np.eye(ndims) * 0.2] * 2)
    data['2,gaussian,std0.2'] = generate_multivariate_normal(samplesz, ndims,
                                                             ncenters=2,
                                                             mus=mus, covs=covs)
    # 2 clusters (gaussians)
    covs = np.array([np.eye(ndims) * 0.1] * 2)
    data['2,gaussian,std0.2'] = generate_multivariate_normal(samplesz, ndims,
                                                             ncenters=2,
                                                             mus=mus, covs=covs)

    # 2 clusters (gaussians elliptical)
    c = .2
    mus = np.ones((2, ndims))*c; mus[0, 0] *= -1
    cov = np.eye(ndims)
    cov[0, 0] = .006
    cov[1, 1] = 1.4
    covs = np.array([cov]*2)
    data['2,gaussian,elliptical'] = generate_multivariate_normal(samplesz, ndims,
                                                               ncenters=2,
                                                               mus=mus,covs=covs)
    return data

##########################################################
def plot_contour_uniform(samplesz, ndims, cs, rs, ax):
    xs = cs[:, 0]
    ys = cs[:, 1]

    xmin = +9999
    xmax = -9999
    ymin = +9999
    ymax = -9999

    for i in range(cs.shape[0]):
        r = rs[i]
        if cs[i, 0] - r < xmin: xmin = cs[i, 0] - r
        if cs[i, 0] + r > xmax: xmax = cs[i, 0] + r

        if cs[i, 1] - r < ymin: ymin = cs[i, 1] - r
        if cs[i, 1] + r > ymax: ymax = cs[i, 1] + r

    xs = np.linspace(xmin, xmax, 500)
    ys = np.linspace(ymin, ymax, 500)

    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros(X.shape)

    for i, c in enumerate(cs):
        x0, y0 = c
        r2 = rs[i]**2
        aux = (X-x0)**2 + (Y-y0)**2
        Z[aux <= r2] = 1

    contours = ax.contour(X, Y, Z)
    return ax

##########################################################
def plot_contour_power(samplesz, ndims, power, mus, ax):
    s = 500

    xmin = +9999
    xmax = -9999
    ymin = +9999
    ymax = -9999

    for i in range(mus.shape[0]):
        r = 1
        if mus[i, 0] - r < xmin: xmin = mus[i, 0] - r
        if mus[i, 0] + r > xmax: xmax = mus[i, 0] + r

        if mus[i, 1] - r < ymin: ymin = mus[i, 1] - r
        if mus[i, 1] + r > ymax: ymax = mus[i, 1] + r

    xs = np.linspace(xmin, xmax, s)
    ys = np.linspace(ymin, ymax, s)

    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    coords = np.zeros((s*s, 2), dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coords[i*s + j, :] = np.array([X[i, j], Y[i, j]])
    Z = np.zeros(X.shape)

    dists = np.zeros(coords.shape[0])
    for i in range(mus.shape[0]):
        mu = mus[0, :]
        dists += cdist(coords, np.array([mu])).flatten()

    Z = np.reshape(dists, X.shape)
    Z = 100 - Z**power
    Z[Z<0] = 0
    contours = ax.contour(X, Y, Z)
    return ax

##########################################################
def plot_contour_gaussian(samplesz, ndims, mus, covs, ax):
    s = 500
    xs = mus[:, 0]
    ys = mus[:, 1]

    xmin = np.min(mus[:, 0]) - 5
    xmax = np.max(mus[:, 0]) + 5
    ymin = np.min(mus[:, 1]) - 5
    ymax = np.max(mus[:, 1]) + 5

    # def f(mus, cov)

    xs = np.linspace(xmin, xmax, s)
    ys = np.linspace(ymin, ymax, s)

    X, Y = np.meshgrid(xs, ys)
    coords = np.zeros((s*s, 2), dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coords[i*s + j, :] = np.array([X[i, j], Y[i, j]])
    Z = np.zeros(X.shape)

    for cind in range(coords.shape[0]):
        i = cind // s
        j = cind % s
        for mind in range(mus.shape[0]):
            Z[i, j] += multivariate_normal(coords[cind, :], mus[mind, :],
                                           covs[mind, :, :])

    contours = ax.contour(X, Y, Z)
    return ax

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
def generate_relevance_distrib_all(data, metricarg, linkagemeths, nrealizations, outdir):
    minnclusters = 2
    minrelsize = 0.3
    nrows = len(data.keys())
    ncols = 2
    samplesz = data[list(data.keys())[0]].shape[0]
    ndims = data[list(data.keys())[0]].shape[1]
    minclustsize = int(minrelsize * samplesz)

    gtruths = compute_gtruth_vectors(data, nrealizations)
    info('Nrealizations:{}, Samplesize:{}, min nclusters:{}, min clustsize:{}'.\
         format(nrealizations, samplesz, minnclusters, minclustsize))


    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5),
                           squeeze=False)

    # plt.tight_layout(pad=5)
    fig.suptitle('Sample size:{}, minnclusters:{}, min clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize),
                 fontsize='x-large', y=0.9)

    # Scatter plot
    for i, distrib in enumerate(data):
        d = data[distrib]
        ax[i, 0].scatter(d[:, 0], d[:, 1])

        # fig = plt.figure(figsize=(8,8))

        # ax = fig.gca()
        # xx = d[:, 0]
        # yy = d[:, 1]
        # xmin = np.min(d[:, 0])
        # xmax = np.max(d[:, 0])
        # ymin = np.min(d[:, 1])
        # ymax = np.max(d[:, 1])
        # ax[i, 0].set_xlim(xmin, xmax)
        # ax[i, 0].set_ylim(ymin, ymax)
        # cfset = ax[i, 0].contourf(xx, yy, f, cmap='coolwarm')
        # ax[i, 0].imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        # cset = ax[i, 0].contour(xx, yy, f, colors='k')
        # ax[i, 0].clabel(cset, inline=1, fontsize=10)
        # ax[i, 0].set_xlabel('X')
        # ax[i, 0].set_ylabel('Y')
        # plt.title('2D Gaussian Kernel density estimation')

    rels = {}
    for k in data.keys():
        rels[k] = {l: [[], []] for l in linkagemeths}

    for _ in range(nrealizations): # Compute relevances
        data = generate_data(samplesz, ndims)

        for j, linkagemeth in enumerate(linkagemeths):
            if linkagemeth == 'centroid' or linkagemeth == 'median' or linkagemeth == 'ward':
                metric = 'euclidean'
            else:
                metric = metricarg

            for i, distrib in enumerate(data):
                z = linkage(data[distrib], linkagemeth, metric)
                inc = inconsistent(z)

                clustids, rel = filter_clustering(data[distrib], z, minclustsize,
                                                        minnclusters)
                clustids = np.array(clustids)
                incinds = clustids - samplesz
                rels[distrib][linkagemeth][len(incinds)-1].append(rel)

    v = {}
    for k in data.keys():
        v[k] = {}

    # v = dict((el, np.zeros(2)) for el in data.keys())
    for i, distrib in enumerate(data):
        for linkagemeth in linkagemeths:
            v[distrib][linkagemeth] = np.zeros(2)
            for j, rel in enumerate(rels[distrib][linkagemeth]):
                v[distrib][linkagemeth][j] = np.sum(rel)

    # Compute the difference vector
    diff = {}
    diffnorms = {}
    for k in data.keys():
        diff[k] = dict((el, np.zeros(2)) for el in linkagemeths)
        diffnorms[k] = {}

    # diff = dict((el, np.zeros(2)) for el in data.keys())
    for i, distrib in enumerate(data):
        for j, linkagemeth in enumerate(linkagemeths):
            diff[distrib][linkagemeth] = gtruths[distrib] - v[distrib][linkagemeth]
            diffnorms[distrib][linkagemeth] = np.linalg.norm(diff[distrib][linkagemeth])
    
    winner = {}
    for d in data.keys():
        minvalue = 1000
        for l in linkagemeths:
            if diffnorms[d][l] < minvalue:
                winner[d] = l
                minvalue = diffnorms[d][l]

    palette = np.array([
    [0.0,0.0,0.0],
    [244,109,67],
    [253,174,97],
    [254,224,139],
    [217,239,139],
    [166,217,106],
    [102,189,99],
    [26,152,80],
           ])
    alpha = .7
    palette /= 255.0
    colours = np.zeros((palette.shape[0], 4), dtype=float)
    colours[:, :3] = palette
    colours[:, -1] = alpha
    palette = colours


    nbins = 10
    bins = np.arange(0, 1, 0.05)
    origin = np.zeros(2)
    for i, distrib in enumerate(data): # Plot
            # ys = np.array([gtruths[distrib][1], v[distrib][linkagemeth][1]])
        xs = np.array([gtruths[distrib][0]])
        ys = np.array([gtruths[distrib][1]])

        ax[i, 1].quiver(origin, origin, xs, ys, color=palette[0], width=.01,
                        angles='xy', scale_units='xy', scale=1,
                        label='Gtruth',
                        headwidth=5, headlength=4, headaxislength=3.5, zorder=0)

        for j, linkagemeth in enumerate(linkagemeths):
            xs = np.array([v[distrib][linkagemeth][0]])
            ys = np.array([v[distrib][linkagemeth][1]])

            coords = np.array([v[distrib][linkagemeth][0],
                              v[distrib][linkagemeth][1]])
            ax[i, 1].quiver(origin, origin, xs, ys, color=palette[j+1], width=.01,
                            angles='xy', scale_units='xy', scale=1,
                            # scale=nrealizations, label=linkagemeth,
                            label=linkagemeth,
                            headwidth=5, headlength=4, headaxislength=3.5,
                            zorder=1/np.linalg.norm(coords))

            ax[i, 1].set_xlim(0, nrealizations)
            ax[i, 1].set_ylim(0, nrealizations)

        plt.text(0.5, 0.9, 'winner:{}'.format(winner[distrib]),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize='large', transform = ax[i, 1].transAxes)

        ax[i, 1].set_ylabel('2 clusters', fontsize='medium')
        ax[i, 1].set_xlabel('1 cluster', fontsize='medium')
        ax[i, 1].legend()

    for i, distrib in enumerate(data): # Plot
        ax[i, 0].set_ylabel('{}'.format(distrib), size='x-large')

    plt.savefig(pjoin(outdir, '{}d_{}_vectors.pdf'.format(ndims, samplesz)))

##########################################################
def test_inconsistency():
    data = {}
    data['A'] = np.array([[x] for x in [10, 20, 100, 200, 400, 1000]])
    data['B'] = np.array([[x] for x in [10, 20, 100, 200, 500, 1000]])

    for i, distrib in enumerate(data):
        z = linkage(data[distrib], 'single')

        print(distrib)
        print(z)
        print(inconsistent(z))

        fancy_dendrogram(
        # dendrogram(
            z,
            color_threshold=0,
            truncate_mode=None,
            leaf_rotation=90.,
            leaf_font_size=7.,
            show_contracted=False,
            show_leaf_counts=True,
        )
        plt.text(0.5, 0.9, '{}'.\
                 format(distrib),
                 ha='center', va='center',
                 fontsize=20)
        plt.ylim(0, 700)
        plt.savefig('/tmp/' + distrib + '.png', dpi=180)
        plt.clf()

##########################################################
def generate_dendrograms_all(data, metricarg, linkagemeths, outdir):
    minnclusters = 2
    minrelsize = 0.3
    samplesz = data[list(data.keys())[0]].shape[0]
    ndims = data[list(data.keys())[0]].shape[1]
    minclustsize = int(minrelsize * samplesz)
    nlinkagemeths = len(linkagemeths)
    ndistribs = len(data.keys())
    nrows = ndistribs
    ncols = nlinkagemeths + 1
    fig = plt.figure(figsize=(ncols*5, nrows*5))
    ax = np.array([[None]*ncols]*nrows)

    fig.suptitle('Sample size:{}, minnclusters:{},\nmin clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize), fontsize=24)

    nsubplots = nrows * ncols

    for subplotidx in range(nsubplots):
        i = subplotidx // ncols
        j = subplotidx % ncols

        if ndims == 3 and j == 0: proj = '3d'
        else: proj = None

        ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1, projection=proj)

    for i, k in enumerate(data):
        info(k)
        nclusters = int(k.split(',')[0])
        plot_scatter(data[k], ax[i, 0], ndims)

        for j, l in enumerate(linkagemeths):
            if l == 'centroid' or l == 'median' or l == 'ward':
                metric = 'euclidean'
            else:
                metric = metricarg
            z = linkage(data[k], l, metric)
            clustids, rel = filter_clustering(data[k], z, minclustsize,
                                                    minnclusters)
            plot_dendrogram(z, l, ax[i, j+1], rel, clustids)
            plt.text(0.7, 0.9, '{}, rel:{:.3f}'.format(len(clustids), rel),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=24, transform = ax[i, j+1].transAxes)

    for ax_, col in zip(ax[0, 1:], linkagemeths):
        ax_.set_title(col, size=20)

    for i, k in enumerate(data):
        ax[i, 0].set_ylabel(k, rotation=90, size=24)

    plt.tight_layout()
    plt.savefig(pjoin(outdir, '{}d_{}_dendrograms.pdf'.format(ndims, samplesz)))

##########################################################
def generate_dendrogram_single(data, metric):
    minnclusters = 2
    minrelsize = 0.3
    nrows = len(data.keys())
    ncols = 2
    samplesz = data[list(data.keys())[0]].shape[0]
    ndims = data[list(data.keys())[0]].shape[1]
    minclustsize = int(minrelsize * samplesz)
    ndistribs = len(data.keys())

    nsubplots = nrows * ncols

    if ndims == 3:
        fig = plt.figure(figsize=(ncols*5, nrows*5))
        ax = np.array([[None]*ncols]*nrows)

        for subplotidx in range(nsubplots):
            i = subplotidx // ncols
            j = subplotidx % ncols

            if j == 0:
                ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1, projection='3d')
            else:
                ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1)
    else:
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5), squeeze=False)

    for i, k in enumerate(data):
        info(k)
        nclusters = int(k.split(',')[0])

        z = linkage(data[k], 'ward', metric)
        clustids, rel = filter_clustering(data[k], z, minclustsize,
                                                minnclusters)
        colours = plot_dendrogram(z, 'ward', ax[i, 1], rel, clustids)
        plot_scatter(data[k], ax[i, 0], ndims, colours)

        plt.text(0.7, 0.9, '{}, rel:{:.3f}'.format(len(clustids), rel),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=30, transform = ax[i, 1].transAxes)

    # for ax_, col in zip(ax[0, 1:], linkagemeths):
        # ax_.set_title(col, size=36)

    for i, k in enumerate(data):
        ax[i, 0].set_ylabel(k, rotation=90, size=24)

    fig.suptitle('Sample size:{}, minnclusters:{},\nmin clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize),
                 y=.92, fontsize=32)

    plt.tight_layout()
    plt.savefig('/tmp/{}d-single.pdf'.format(ndims))

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ndims', type=int, default=2,
                        help='Dimensionality of the space')
    parser.add_argument('--samplesz', type=int, default=100, help='Sample size')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.INFO)

    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(0)

    ##########################################################
    nrealizations = 3

    metric = 'euclidean'
    linkagemeths = ['single', 'complete', 'average', 'centroid', 'median', 'ward']
    info('Computing:{}'.format(linkagemeths))
    data = generate_data(args.samplesz, args.ndims)
    # generate_dendrograms_all(data, metric, linkagemeths, args.outdir)
    # generate_dendrogram_single(data, metric)
    generate_relevance_distrib_all(data, metric, linkagemeths, nrealizations, args.outdir)
    # test_inconsistency()

##########################################################
if __name__ == "__main__":
    main()

