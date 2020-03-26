#!/usr/bin/env python3
"""Benchmark on hierarchical clustering methods
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
import os

import numpy as np

import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt; plt.style.use('ggplot')
import matplotlib.cm as cm
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import inconsistent
import scipy.stats as stats
from scipy.spatial.distance import cdist
import pandas as pd

from sklearn import datasets
import imageio

##########################################################
def multivariate_normal(x, mean, cov):
    """P.d.f. of the multivariate normal when the covariance matrix is positive definite.
    Source: wikipedia"""
    ndims = len(mean)
    B = x - mean
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
def plot_dendrogram(z, linkagemeth, ax, lthresh, clustids, palette):
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
    # vividcolors = ['b', 'g', 'r', 'c', 'm']
    vividcolors = palette.copy()

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
        no_labels=True,
        ax=ax,
        link_color_func=lambda k: colors[k],
    )
    if lthresh > 0:
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

        cursize = 0
        aux2 = np.ndarray((0, ndims), dtype=float)
        while True:
            aux = np.random.rand(partsz[i], ndims) * range_ + min_
            dists = cdist(aux, np.array([c]))
            inds = np.where(dists <= r)[0]
            aux = aux[inds, :]
            cursize += aux.shape[0]
            aux2 = np.concatenate((aux2, aux))

            if cursize >= partsz[i]: break

        data[dataind:dataind+partsz[i]] = aux2[:partsz[i]]
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

    ind = 0
    for i in range(ncenters):
        mu = mus[i, :]
        cov = covs[i, :, :]
        x[ind:ind+partsz[i]] = np.random.multivariate_normal(mu, cov, size=partsz[i])
        ind += partsz[i]
    return x

##########################################################
def generate_exponential(samplesz, ndims, mus=[]):
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
        sz = partsz[i]
        xs = np.random.exponential(size=sz)
        ys = np.random.exponential(size=sz)
        xs[np.random.rand(sz) > .5] *= -1
        ys[np.random.rand(sz) > .5] *= -1
        x[ind:ind+sz, 0] = xs[:sz] + mu[0]
        x[ind:ind+sz, 1] = ys[:sz] + mu[1]
        ind += sz
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
        sz = partsz[i]
        xs = 1 - np.random.power(a=power+1, size=sz)
        ys = 1 - np.random.power(a=power+1, size=sz)
        xs[np.random.rand(sz) > .5] *= -1
        ys[np.random.rand(sz) > .5] *= -1
        x[ind:ind+sz, 0] = xs[:sz] + mu[0]
        x[ind:ind+sz, 1] = ys[:sz] + mu[1]
        ind += sz
    return x

##########################################################
def plot_scatter(x, ax, ndims, coloursarg=None):
    if ndims == 3:
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    else:
        ax.scatter(x[:,0], x[:,1], c=coloursarg)

##########################################################
def generate_data(samplesz, ndims):
    """Synthetic data

    Args:
    n(int): size of each sample

    Returns:
    list of np.ndarray: each element is a nx2 np.ndarray
    """

    info('Generating data...')
    data = {}

    # fig, ax = plt.subplots(2)
    # mus = np.zeros((1, ndims))
    # ax[0] = plot_contour_power(samplesz, ndims, 1, mus, ax[0])
    # ax[1] = plot_contour_power(samplesz, ndims, 5, mus, ax[1])
    # plt.show()

    mu = np.zeros((1, ndims))
    r = np.array([.9])
    data['1,uniform,rad0.9'] = generate_uniform(samplesz, ndims, mu, r)
 
    # 1 cluster (linear)
    mus = np.zeros((1, ndims))
    data['1,linear'] = generate_power(samplesz, ndims, power=1, mus=mus)

    # 1 cluster (power)
    mus = np.zeros((1, ndims))
    data['1,quadratic'] = generate_power(samplesz, ndims, power=2, mus=mus)

    # 1 cluster (gaussian)
    mus = np.zeros((1, ndims))
    covs = np.array([np.eye(ndims) * 0.1])
    data['1,gaussian'] = generate_multivariate_normal(samplesz, ndims, ncenters=1,
                                                      mus=mus, covs=covs)

    # 1 cluster (exponential)
    mus = np.zeros((1, ndims))
    data['1,exponential'] = generate_exponential(samplesz, ndims, mus=mus)

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
    data['2,gaussian,std0.1'] = generate_multivariate_normal(samplesz, ndims,
                                                             ncenters=2,
                                                             mus=mus, covs=covs)

    # 2 clusters (gaussians elliptical)
    mus = np.zeros((2, ndims));
    mus[0, 0] = -.3
    mus[1, 0] = +.3
    cov = np.eye(ndims)
    cov[0, 0] = .01
    covs = np.array([cov]*2)
    data['2,gaussian,elliptical'] = generate_multivariate_normal(samplesz, ndims,
                                                               ncenters=2,
                                                               mus=mus,covs=covs)
    # d = data['2,gaussian,elliptical']
    # plt.scatter(d[:, 0], d[:, 1])
    # plt.savefig('/tmp/out.pdf')
    # input()

    # data['4,iris'] = datasets.load_iris().data
    return data

##########################################################
def mesh_xy(min, max, s):
    xs = np.linspace(-1, +1, s)
    ys = np.linspace(-1, +1, s)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    return X, Y, Z

##########################################################
def plot_contour_uniform(mus, rs, s, ax, cmap, linewidth):
    X, Y, Z = mesh_xy(-1.0, +1.0, s)

    for i, c in enumerate(mus):
        x0, y0 = c
        r2 = rs[i]**2
        aux = (X-x0)**2 + (Y-y0)**2
        Z[aux <= r2] = 1

    contours = ax.contour(X, Y, Z, levels=1, cmap=cmap, linewidths=linewidth)
    return ax

##########################################################
def plot_contour_power(ndims, power, mus, s, ax, cmap, linewidth):
    X, Y, Z = mesh_xy(-1.0, +1.0, s)
    coords = np.zeros((s*s, 2), dtype=float)
    Zflat = np.zeros(s*s, dtype=float)
    epsilon = 1.2

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coords[i*s + j, :] = np.array([X[i, j], Y[i, j]])

    for i in range(mus.shape[0]):
        mu = mus[0, :]
        d = cdist(coords, np.array([mu])).flatten()
        # Zflat += -d**power
        Zflat += (1 / (d+epsilon)) ** power

    Z = np.reshape(Zflat, X.shape)
    contours = ax.contour(X, Y, Z, cmap=cmap, levels=3, linewidths=linewidth)
    return ax

##########################################################
def plot_contour_exponential(ndims, mus, s, ax, cmap, linewidth):
    X, Y, Z = mesh_xy(-1.0, +1.0, s)
    coords = np.zeros((s*s, 2), dtype=float)
    Zflat = np.zeros(s*s, dtype=float)
    epsilon = 0.5

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coords[i*s + j, :] = np.array([X[i, j], Y[i, j]])

    for i in range(mus.shape[0]):
        mu = mus[0, :]
        d = cdist(coords, np.array([mu])).flatten()
        # Zflat += -np.exp(d)
        Zflat += np.exp(1 / (d+epsilon))

    Z = np.reshape(Zflat, X.shape)
    contours = ax.contour(X, Y, Z, cmap=cmap, levels=3, linewidths=linewidth)
    return ax
##########################################################
def plot_contour_gaussian(ndims, mus, covs, s, ax, cmap, linewidth):
    X, Y, Z = mesh_xy(-1.0, +1.0, s)
    coords = np.zeros((s*s, 2), dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coords[i*s + j, :] = np.array([X[i, j], Y[i, j]])

    for cind in range(coords.shape[0]):
        i = cind // s
        j = cind % s
        for mind in range(mus.shape[0]):
            Z[i, j] += multivariate_normal(coords[cind, :], mus[mind, :],
                                           covs[mind, :, :])

    contours = ax.contour(X, Y, Z, cmap=cmap, levels=3, linewidths=linewidth)
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
def export_individual_axis(ax, fig, labels, outdir, pad=0.3, prefix=''):
    n = ax.shape[0]*ax.shape[1]
    for k in range(n):
        i = k // ax.shape[1]
        j = k  % ax.shape[1]
        coordsys = fig.dpi_scale_trans.inverted()
        extent = ax[i, j].get_window_extent().transformed(coordsys)
        fig.savefig(pjoin(outdir, prefix + labels[k] + '.png'),
                      bbox_inches=extent.expanded(1+pad, 1+pad))

##########################################################
def plot_contours(labels, outdir, icons=False):
    ndims = 2
    s = 500
    nrows = 2
    ncols = 5
    cmap = 'Blues'

    if icons:
        figscale = .3
        linewidth = 1
    else:
        figscale = 5
        linewidth = 3

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*figscale, nrows*figscale), squeeze=False)
    # Contour plots
    mu = np.array([[0, 0]])
    r = np.array([.9])
    plot_contour_uniform(mu, r, s, ax[0, 0], cmap, linewidth) # 1 uniform
 
    mus = np.zeros((1, ndims))

    plot_contour_power(ndims, 1, mus, s, ax[0, 1], cmap, linewidth) # 1 linear

    plot_contour_power(ndims, 2, mus, s, ax[0, 2], cmap, linewidth) # 1 power

    covs = np.array([np.eye(ndims) * 0.15]) # 1 gaussian
    plot_contour_gaussian(ndims, mus, covs, s, ax[0, 3], cmap, linewidth)

    mus = np.zeros((1, ndims))
    plot_contour_exponential(ndims, mus, s, ax[0, 4], cmap, linewidth) # 1 exponential

    c = 0.7
    mus = np.ones((2, ndims))*c; mus[1, :] *= -1
    rs = np.ones(2) * .9
    plot_contour_uniform(mus, rs, s, ax[1, 0], cmap, linewidth) # 2 uniform

    rs = np.ones(2) * .5
    plot_contour_uniform(mus, rs, s, ax[1, 1], cmap, linewidth) # 2 uniform

    covs = np.array([np.eye(ndims) * 0.2] * 2)
    plot_contour_gaussian(ndims, mus, covs, s, ax[1, 2], cmap, linewidth) # 2 gaussians

    covs = np.array([np.eye(ndims) * 0.1] * 2)
    plot_contour_gaussian(ndims, mus, covs, s, ax[1, 3], cmap, linewidth) # 2 gaussians

    mus = np.zeros((2, ndims));
    mus[0, 0] = -.3
    mus[1, 0] = +.3
    cov = np.eye(ndims)
    cov[0, 0] = .01
    covs = np.array([cov]*2)
    plot_contour_gaussian(ndims, mus, covs, s, ax[1, 4], cmap, linewidth) # 2 gaussians ellip


    plt.tight_layout(pad=4)
    plt.savefig(pjoin(outdir, 'contour_all_{}d.pdf'.format(ndims)))
    for i in range(ax[:].shape[0]):
        for j in range(ax[:].shape[1]):
            if icons:
                ax[i, j].set_yticks([])
                ax[i, j].grid(False)
                ax[i, j].set_yticklabels([])
                ax[i, j].set_xticklabels([])
            else:
                ax[i, j].set_xticks([-1.0, 0, +1.0])
                ax[i, j].set_yticks([-1.0, 0, +1.0])
                ax[i, j].set_xticks([])

    if icons: pref = 'icon_'
    else: pref = 'contour_'
    export_individual_axis(ax, fig, labels, outdir, .0, pref)

##########################################################
def hex2rgb(hexcolours, alpha=None):
    rgbcolours = np.zeros((len(hexcolours), 3), dtype=int)
    for i, h in enumerate(hexcolours):
        rgbcolours[i, :] = np.array([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])

    if alpha != None:
        aux = np.zeros((len(hexcolours), 4), dtype=float)
        aux[:, :3] = rgbcolours / 255.0
        aux[:, -1] = .7 # alpha
        rgbcolours = aux

    return rgbcolours

##########################################################
def generate_relevance_distrib_all(data, metricarg, linkagemeths, nrealizations,
                                   palettehex, outdir):
    info('Computing relevances...')
    s = 500
    cmap = 'Blues'
    minnclusters = 2
    minrelsize = 0.3
    nrows = len(data.keys())
    ncols = 1
    samplesz = data[list(data.keys())[0]].shape[0]
    ndims = data[list(data.keys())[0]].shape[1]
    minclustsize = int(minrelsize * samplesz)

    gtruths = compute_gtruth_vectors(data, nrealizations)
    info('Nrealizations:{}, Samplesize:{}, min nclusters:{}, min clustsize:{}'.\
         format(nrealizations, samplesz, minnclusters, minclustsize))


    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4),
                           squeeze=False)

    # plt.tight_layout(pad=5)
    rels = {}
    for k in data.keys():
        rels[k] = {l: [[], []] for l in linkagemeths}

    for r in range(nrealizations): # Compute relevances
        info('Realization {:02d}'.format(r))
        data = generate_data(samplesz, ndims)

        for j, linkagemeth in enumerate(linkagemeths):
            if linkagemeth == 'centroid' or linkagemeth == 'median' or linkagemeth == 'ward':
                metric = 'euclidean'
            else:
                metric = metricarg

            for i, distrib in enumerate(data):
                try:
                    z = linkage(data[distrib], linkagemeth, metric)
                except Exception as e:
                    print(data[distrib], linkagemeth, metric)
                    np.save('/tmp/foo.npy', data[distrib])
                    print(e)
                    raise(e)

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

    df = pd.DataFrame.from_dict(diffnorms, orient='index')
    df['dim'] = pd.Series([ndims for x in range(len(df.index))], index=df.index)
    df.to_csv(pjoin(outdir, 'results.csv'), sep='|', index_label='distrib')

    fh = open(pjoin(outdir, 'raw.csv'), 'w')
    fh.write('distrib|linkagemeth|realiz|relev1|relev2\n')
    for d in data.keys():
        for l in linkagemeths:
            i = 0
            rels1 = rels[d][l][0]
            for r in rels1:
                fh.write(('{}|{}|{}|{}|0.0\n'.format(d, l, i, r)))
                i += 1

            rels2 = rels[d][l][1]
            for r in rels2:
                fh.write(('{}|{}|{}|0.0|{}\n'.format(d, l, i, r)))
                i += 1
    fh.close()

    if ndims > 2: return

    palette = hex2rgb(palettehex, alpha=.8)

    nbins = 10
    bins = np.arange(0, 1, 0.05)
    origin = np.zeros(2)
    for i, distrib in enumerate(data): # Plot
            # ys = np.array([gtruths[distrib][1], v[distrib][linkagemeth][1]])
        xs = np.array([gtruths[distrib][0]])
        ys = np.array([gtruths[distrib][1]])

        ax[i, 0].quiver(origin, origin, xs, ys, color='#000000', width=.01,
                        angles='xy', scale_units='xy', scale=1,
                        label='Gtruth',
                        headwidth=5, headlength=4, headaxislength=3.5, zorder=3)

        for j, linkagemeth in enumerate(linkagemeths):
            xs = np.array([v[distrib][linkagemeth][0]])
            ys = np.array([v[distrib][linkagemeth][1]])

            coords = np.array([v[distrib][linkagemeth][0],
                              v[distrib][linkagemeth][1]])
            ax[i, 0].quiver(origin, origin, xs, ys, color=palette[j], width=.01,
                            angles='xy', scale_units='xy', scale=1,
                            # scale=nrealizations, label=linkagemeth,
                            label=linkagemeth,
                            headwidth=5, headlength=4, headaxislength=3.5,
                            zorder=1/np.linalg.norm(coords)+3)

            ax[i, 0].set_xlim(0, nrealizations)
            ax[i, 0].set_ylim(0, nrealizations)
            # ax[i, 0].set_axisbelow(True)

        # plt.text(0.5, 0.9, 'winner:{}'.format(winner[distrib]),
                 # horizontalalignment='center', verticalalignment='center',
                 # fontsize='large', transform = ax[i, 1].transAxes)

        ax[i, 0].set_ylabel('Sum of relevances of 2 clusters', fontsize='medium')
        ax[i, 0].set_xlabel('Sum of relevances of 1 cluster', fontsize='medium')
        ax[i, 0].legend()

    plt.tight_layout(pad=4)
    export_individual_axis(ax, fig, list(data.keys()), outdir, 0.36, 'vector_')
    fig.suptitle('Sample size:{}, minnclusters:{}, min clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize),
                 fontsize='x-large', y=0.98)


    for i, distrib in enumerate(data): # Plot
        ax[i, 0].set_ylabel('{}'.format(distrib), size='x-large')

    plt.savefig(pjoin(outdir, 'vector_{}d_{}.pdf'.format(ndims, samplesz)))

##########################################################
def test_inconsistency():
    data = {}
    data['A'] = np.array([[x] for x in [10, 20, 100, 200, 400, 1000]])
    data['B'] = np.array([[x] for x in [10, 20, 100, 200, 500, 1000]])

    for i, distrib in enumerate(data):
        z = linkage(data[distrib], 'single')


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
def generate_dendrograms_all(data, metricarg, linkagemeths, palette, outdir):
    info('Generating dendrograms...')
    minnclusters = 2
    minrelsize = 0.3
    samplesz = data[list(data.keys())[0]].shape[0]
    ndims = data[list(data.keys())[0]].shape[1]
    minclustsize = int(minrelsize * samplesz)
    nlinkagemeths = len(linkagemeths)
    ndistribs = len(data.keys())
    nrows = ndistribs
    ncols = nlinkagemeths + 1
    figscale=5
    fig = plt.figure(figsize=(ncols*figscale, nrows*figscale))
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
        nclusters = int(k.split(',')[0])
        plot_scatter(data[k], ax[i, 0], ndims, palette[1])

        for j, l in enumerate(linkagemeths):
            if l == 'centroid' or l == 'median' or l == 'ward':
                metric = 'euclidean'
            else:
                metric = metricarg
            z = linkage(data[k], l, metric)
            clustids, rel = filter_clustering(data[k], z, minclustsize, minnclusters)
            # plot_dendrogram(z, l, ax[i, j+1], rel, clustids, palette)
            plot_dendrogram(z, l, ax[i, j+1], 0, clustids, ['k']*10)

            if len(clustids) == 1:
                text = 'rel:({:.3f}, 0.0)'.format(rel)
            else:
                text = 'rel:(0.0, {:.3f})'.format(rel)

            # plt.text(0.6, 0.9, text,
                     # horizontalalignment='center', verticalalignment='center',
                     # fontsize=24, transform = ax[i, j+1].transAxes)

    # for ax_, col in zip(ax[0, 1:], linkagemeths):
        # ax_.set_title(col, size=20)

    for i, k in enumerate(data):
        ax[i, 0].set_ylabel(k, rotation=90, size=24)

    export_individual_axis(ax[0, 1:].reshape(len(linkagemeths), 1), fig, linkagemeths,
                           outdir, 0.2, 'dendrogram_uniform_')
    plt.tight_layout(pad=4)

    plt.savefig(pjoin(outdir, 'dendrogram_all_{}d_{}.pdf'.format(ndims, samplesz)))

##########################################################
def plot_article_uniform_distribs_scale(data, palette, outdir):
    info('Plotting uniform bimodal distrib...')
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
    k = '2,uniform,rad0.9'
    ndims = 2
    plot_scatter(data[k], ax[0, 0], ndims, palette[1])

    mycolour = 'k'
    c = (0, 0)
    r = 2
    m = r * np.sqrt(2) / 2
    circle = Circle(c, r, facecolor='none',
                    edgecolor=mycolour, linestyle='-',
                    linewidth=2, alpha=0.5,
                    )
    ax[0, 0].add_patch(circle)
    origin = np.zeros(2)
    # ax[0, 0].plot([0, r], [0, 0], 'k--', linewidth=2, alpha=.5)
    ax[0, 0].quiver(origin, origin, [-1.41], [1.41], color=mycolour, width=.008,
              angles='xy', scale_units='xy', scale=1, alpha=.3,
              headwidth=5, headlength=4, headaxislength=3.5, zorder=3)
    
    nnoise = 60
    noisex = np.random.rand(nnoise) * 4 + (-2)
    noisey = np.random.rand(nnoise) * 4 + (-2)
    ax[0, 0].scatter(noisex, noisey, c=palette[1]) # noise points
    ax[0, 0].scatter(noisex, noisey, c=palette[1]) # noise points
    ax[0, 0].set_xticks([-2.0, -1.0, 0, +1.0, +2.0])
    ax[0, 0].set_yticks([-2.0, -1.0, 0, +1.0, +2.0])

    export_individual_axis(ax, fig, [k], outdir, 0.3, 'points_')

##########################################################
def generate_dendrogram_single(data, metric, palettehex, outdir):
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

        z = linkage(data[k], 'single', metric)
        clustids, rel = filter_clustering(data[k], z, minclustsize,
                                                minnclusters)
        colours = plot_dendrogram(z, 'single', ax[i, 1], rel, clustids, palettehex)
        plot_scatter(data[k], ax[i, 0], ndims, colours)

        if len(clustids) == 1:
            text = 'rel: ({:.3f}, 0)'.format(rel)
        else:
            text = 'rel: (0, {:.3f})'.format(rel)
        plt.text(0.7, 0.9, text,
        # plt.text(0.7, 0.9, '{}, rel:{:.3f}'.format(len(clustids), rel),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20, transform = ax[i, 1].transAxes)

    # for ax_, col in zip(ax[0, 1:], linkagemeths):
        # ax_.set_title(col, size=36)

    for i, k in enumerate(data):
        ax[i, 0].set_ylabel(k, rotation=90, size=24)

    fig.suptitle('Sample size:{}, minnclusters:{},\nmin clustsize:{}'.\
                 format(samplesz, minnclusters, minclustsize),
                 y=.92, fontsize=32)

    plt.savefig(pjoin(outdir, '{}d-single.pdf'.format(ndims)))

##########################################################
def plot_article_quiver(palettehex, outdir):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    orig = np.zeros(2)
    v1 = np.array([0.6, 1.8])
    v2 = np.array([3.0, 0.0])
    v3 = v2 - v1
    delta = .01

    ax.quiver(orig[0], orig[1], v1[0], v1[1], color=palettehex[0],
              width=.015, angles='xy', scale_units='xy', scale=1,
              headwidth=4, headlength=4, headaxislength=3, zorder=3)

    ax.quiver(orig[0], orig[1]+delta, v2[0]-delta, v2[1],
              color='#000000', alpha=.7,
              width=.015, angles='xy', scale_units='xy', scale=1,
              headwidth=4, headlength=4, headaxislength=3, zorder=3)

    ax.quiver(v1[0] + delta, v1[1] - delta, v3[0]-delta, v3[1]+delta,
              color='#999999', alpha=1.0,
              width=.007, angles='xy', scale_units='xy', scale=1,
              headwidth=5, headlength=8, headaxislength=5, zorder=3)

    ax.scatter([0.6], [1.8], s=2, c='k')
    # plt.text(0.32, 0.9, '(0.6, 1.8)',
             # horizontalalignment='center', verticalalignment='center',
             # fontsize='large', transform = ax.transAxes)
    plt.text(0.16, 0.47, 'r',
             horizontalalignment='center', verticalalignment='center',
             color=palettehex[0], style='italic',
             fontsize='x-large', transform = ax.transAxes)
    plt.text(0.45, 0.08, 'g',
             horizontalalignment='center', verticalalignment='center', style='italic',
             fontsize='x-large', transform = ax.transAxes)
    plt.text(0.75, 0.3, 'e',
             horizontalalignment='center', verticalalignment='center', style='italic',
             color='#666666',
             fontsize='x-large', transform = ax.transAxes)
    ax.set_ylabel('Relevance of 2 clusters', fontsize='medium')
    ax.set_xlabel('Relevance of 1 cluster', fontsize='medium')
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2)
    plt.tight_layout(pad=1)

    plt.savefig(pjoin(outdir, 'arrow_dummy.pdf'))

##########################################################
def plot_parallel(df, colours, ax, fig):
    dim = df.dim[0]
    df = df.T.reset_index()
    df = df[df['index'] != 'dim']

    ax = pd.plotting.parallel_coordinates(
        df, 'index',
        axvlines_kwds={'visible':True, 'color':np.ones(3)*.6,
                       'linewidth':4, 'alpha': 0.9, },
        ax=ax, linewidth=4, alpha=0.9,
        color = colours,
    )
    ax.yaxis.grid(False)
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_xticklabels([])
    ax.set_xlim(-.5, 10)
    ax.set_ylabel('Accumulated error', fontsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(
        fontsize=15,
        loc=[.55, .55],
    )

    ax.tick_params(bottom="off")
    ax.tick_params(axis='x', length=0)

    # axicon = fig.add_axes([0.4,0.4,0.1,0.1])
    # axicon.imshow(np.random.rand(100,100))
    # axicon.set_xticks([])
    # axicon.set_yticks([])

    trans = blended_transform_factory(fig.transFigure, ax.transAxes) # separator
    line = Line2D([0, .98], [-.02, -.02], color='k', transform=trans)
    fig.lines.append(line)

##########################################################
def include_icons(iconpaths, fig):
    for i, iconpath in enumerate(iconpaths):
        sign = 0.015*(-1) ** i
        im = imageio.imread(iconpath)
        newax = fig.add_axes([0.21+i*.0723, 0.78+sign, 0.08, 0.2], anchor='NE', zorder=-1)
        newax.imshow(im, aspect='equal')
        newax.axis('off')

##########################################################
def plot_parallel_all(results, outdir):
    if not os.path.isdir(outdir): os.mkdir(outdir)

    colours = cm.get_cmap('tab10')(np.linspace(0, 1, 6))
    df = pd.read_csv(results, sep='|')
    dims = np.unique(df.dim)

    figscale = 5
    fig, axs = plt.subplots(len(dims), 1, figsize=(1.5*figscale, len(dims)*figscale),
                            squeeze=False)

    for i, dim in enumerate(dims):
        slice = df[df.dim == dim]
        # print('slice:{}'.format(slice))
        slice = slice.set_index('distrib')
        plot_parallel(slice, colours, axs[i, 0], fig)

    # plt.tight_layout(rect=(0.1, 0, 1, 1))
    plt.tight_layout(rect=(0.1, 0, 1, .96), h_pad=-2)
    for i, dim in enumerate(dims):
        plt.text(-0.2, .5, '{}-D'.format(dim),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize='xx-large',
                 transform = axs[i, 0].transAxes
                 )

    filenames = [
        'icon_1,uniform,rad0.9.png',
        'icon_1,linear.png',
        'icon_1,quadratic.png',
        'icon_1,gaussian.png',
        'icon_1,exponential.png',
        'icon_2,uniform,rad0.9.png',
        'icon_2,uniform,rad0.5.png',
        'icon_2,gaussian,std0.2.png',
        'icon_2,gaussian,std0.1.png',
        'icon_2,gaussian,elliptical.png',
    ]
    iconpaths = [ pjoin('/tmp/', f) for f in filenames ]

    include_icons(iconpaths, fig)

    plt.savefig(pjoin(outdir, 'error_all.pdf'))

##########################################################
def count_method_ranking(resultspath, linkagemeths, outdir, linkagemeth='single'):
    print('linkagemeth:{}'.format(linkagemeth))
    df = pd.read_csv(resultspath, sep='|')
    methidx = np.where(np.array(linkagemeths) == linkagemeth)[0][0]

    data = []
    for i, row in df.iterrows():
        values = row[linkagemeths].values
        # print('values:{}'.format(values))
        # input()
        sortedidx = np.argsort(values)
        methrank = np.where(sortedidx == methidx)[0][0]

        # print('##########################################################')
        # print(values)
        # print('dim:{}'.format(row.dim))
        # print('i:{}'.format(i))
        # print('sortedidx:{}'.format(sortedidx))
        # print('methrank:{}'.format(methrank))
        # input()
        data.append([row.distrib, row.dim, methrank])

    methdf = pd.DataFrame(data,
                          columns='distrib,dim,methrank'.split(','))
    # print('methdf:{}'.format(methdf))
    # input()
    ndistribs = len(np.unique(methdf.distrib))

    for d in np.unique(methdf.dim):
        # print(methdf[methdf.dim == d])
        filtered1 = methdf[methdf.dim == d][:5]
        filtered1 = filtered1[(filtered1.methrank < 3)]

        filtered2 = methdf[methdf.dim == d][5:]
        filtered2 = filtered2[(filtered2.methrank < 3)]

        print('dim:{}\t{}/{}\t{}/{}'.format(d, filtered1.shape[0], 5,
                                     filtered2.shape[0], 5))
##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ndims', type=int, default=2,
                        help='Dimensionality of the space')
    parser.add_argument('--samplesz', type=int, default=100, help='Sample size')
    parser.add_argument('--nrealizations', type=int, default=400, help='Sample size')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.INFO)

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(0)

    metric = 'euclidean'
    linkagemeths = ['single', 'complete', 'average', 'centroid', 'median', 'ward']
    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # palettehex = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    info('Linkage methods:{}'.format(linkagemeths))

    data = generate_data(args.samplesz, args.ndims)
    # generate_dendrograms_all(data, metric, linkagemeths, palettehex, args.outdir)
    # generate_dendrogram_single(data, metric, palettehex, args.outdir)
    # generate_relevance_distrib_all(data, metric, linkagemeths,
                                   # args.nrealizations, palettehex,
                                   # args.outdir)
    # return
    # plot_contours(list(data.keys()), args.outdir)
    # plot_contours(list(data.keys()), args.outdir, True)
    # plot_article_uniform_distribs_scale(data, palettehex, args.outdir)
    # plot_article_quiver(palettehex, args.outdir)
    resultspath = '/home/dufresne/20200326-hieclust.csv'

    # plot_parallel_all(resultspath, args.outdir)
    count_method_ranking(resultspath, linkagemeths, args.outdir, 'single')
    count_method_ranking(resultspath, linkagemeths, args.outdir, 'ward')
    # test_inconsistency()

##########################################################
if __name__ == "__main__":
    main()

