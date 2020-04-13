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
from scipy.stats import pearsonr
import pandas as pd

from sklearn import datasets
import imageio
import scipy
import inspect
import igraph

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
def plot_dendrogram(z, linkagemeth, ax, avgheight, maxheight, clustids, palette, outliers):
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
    # dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
    maxh = np.max(dists)
    dists = dists / maxh
    z[:, 2] = dists
    n = z.shape[0] + 1
    colors = (2 * n - 1) * ['k']
    # vividcolors = ['b', 'g', 'r', 'c', 'm']
    vividcolors = palette.copy()

    for clustid in clustids:
        c = vividcolors.pop()
        links = get_descendants(z, n, clustid)
        for l in links: colors[l]  = c

    c = vividcolors.pop()
    ancestors = []
    for outlier in outliers:
        ancestors.extend(get_ancestors(outlier, z))
    ancestors.extend(outliers)

    for anc in ancestors:
        colors[anc]  = 'b'

    epsilon = 0.0000
    dendrogram(
        z,
        color_threshold=avgheight+epsilon,
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
    if avgheight > 0:
        ax.axhline(y=avgheight/maxh, linestyle='--')
    if maxheight > 0:
        ax.axhline(y=maxheight/maxh, linestyle='--', c='b', alpha=.5)
    return colors[:n]

##########################################################
def get_points_inside_circle(x, c0, r):
    dists = cdist(x, [c0])
    inds = np.where(dists <= r)[0]
    inside = x[inds, :]
    return inside

##########################################################
def get_random_sample(x, npartitions, samplesz):
    ind = np.random.choice(range(x.shape[0]), size=samplesz, replace=False)
    counts = np.zeros(npartitions, dtype=int)
    for k in range(npartitions):
        counts[k] = np.sum( (ind >= k*samplesz) & (ind < (k+1)*samplesz))
    return x[sorted(ind)], counts

##########################################################
def generate_uniform(samplesz, ndims, mus, rads):
    ncenters = len(mus)

    # x is initialized with ncenters times the requested size
    x = np.zeros((samplesz * ncenters, ndims))

    for i in range(ncenters):
        min_ = mus[i] - rads[i]
        range_ = 2 * rads[i]

        aux = np.ndarray((0, ndims), dtype=float)
        while aux.shape[0] < samplesz:
            aux2 = np.random.rand(samplesz, ndims) * range_ + min_
            aux2 = get_points_inside_circle(aux2, mus[i], rads[i])
            aux = np.concatenate((aux, aux2))
        x[i*samplesz: (i+1)*samplesz] = aux[:samplesz]

    return get_random_sample(x, ncenters, samplesz)

##########################################################
def generate_multivariate_normal(samplesz, ndims, mus, covs=[]):
    ncenters = len(mus)
    x = np.ndarray((samplesz*ncenters, ndims), dtype=float)

    for i in range(ncenters):
        cov = covs[i, :, :]

        aux = np.random.multivariate_normal(mus[i], cov, size=samplesz)
        # aux = get_points_inside_circle(aux, c, rs[i])
        x[i*samplesz: (i+1)*samplesz] = aux[:samplesz]

    return get_random_sample(x, ncenters, samplesz)

##########################################################
def generate_exponential(samplesz, ndims, mus, rads):
    ncenters = len(mus)
    x = np.ndarray((samplesz*ncenters, ndims), dtype=float)

    for i in range(ncenters):
        aux = np.random.exponential(rads[i], size=(samplesz, ndims))
        aux *= random_sign((samplesz, ndims))
        # aux = get_points_inside_circle(aux, mu, 1)
        x[i*samplesz: (i+1)*samplesz] = aux[:samplesz] + mus[i]

    return get_random_sample(x, ncenters, samplesz)

##########################################################
def random_sign(sampleshape):
    return (np.random.rand(*sampleshape) > .5).astype(int)*2 - 1

##########################################################
def generate_power(samplesz, ndims, power, mus, rads):
    ncenters = len(mus)
    x = np.ndarray((samplesz*ncenters, ndims), dtype=float)

    for i in range(ncenters):
        aux = rads[i] * (1 - np.random.power(a=power+1, size=(samplesz, ndims)))
        aux *= random_sign((samplesz, ndims))
        # aux = get_points_inside_circle(aux, mu, 1)
        x[i*samplesz: (i+1)*samplesz] = aux[:samplesz] + mus[i]


    return get_random_sample(x, ncenters, samplesz)

##########################################################
def calculate_alpha(x, partsz):
    """Print alpha value

    Args:
    x(np.ndarray): numpy array containing data
    partsz(np.ndarray): np array containing the partition sizes

    Returns:
    float: alpha value
    """

    info(inspect.stack()[0][3] + '()')
    d1 = x[:partsz[0]]
    d2 = x[partsz[0]:]
    k = np.linalg.norm(np.mean(d1, 0))
    d = np.mean([np.linalg.norm(np.mean(d1, 0)), np.linalg.norm(np.mean(d2, 0))])
    stdavg = np.mean([np.std(d1), np.std(d2)])

    alpha = 2 * d / stdavg
    return alpha

##########################################################
def shift_clusters(x, partsz, alpha):
    """Shift cluster accordign to
    alpha = 2*d / std
    where d is the distance between the cluster and the origin
    Resulting clusters are symmetric wrt the origin

    Args:
    x(np.ndarray): numpy array containing data
    alpha(float): parameter for shifting the clusters

    Returns:
    np.ndarray: shifted cluters
    """
    ncenters = len(partsz)
    d1 = x[:partsz[0]]
    d2 = x[partsz[0]:]
    stdavg = np.mean([np.std(d1), np.std(d2)])

    d = alpha * stdavg / 2.0
    mu = d / np.sqrt(ncenters)
    shifted = x.copy()

    deltamu = (-mu) - np.mean(d1)
    shifted[:partsz[0]] += deltamu
    deltamu = (mu) - np.mean(d2)
    shifted[partsz[0]:] += deltamu
    # calculate_alpha(shifted, partsz)
    return shifted

##########################################################
def generate_data(samplesz, ndims):
    """Synthetic data

    Args:
    n(int): size of each sample

    Returns:
    list of np.ndarray: each element is a nx2 np.ndarray
    """

    info(inspect.stack()[0][3] + '()')

    data = {}
    partsz = {}

    mu = np.zeros((1, ndims))

    k = '1,uniform'
    r = np.array([1])
    data[k], partsz[k] = generate_uniform(samplesz, ndims, mu, r)
 
    k = '1,gaussian'
    cov = np.array([np.eye(ndims)])*.3
    data[k], partsz[k] = generate_multivariate_normal(samplesz, ndims, mu, cov)

    k = '1,quadratic'
    data[k], partsz[k] = generate_power(samplesz, ndims, 2, mu, np.array([1])*1)

    k = '1,exponential'
    data[k], partsz[k] = generate_exponential(samplesz, ndims, mu, np.ones(1)*.3)

    mus = np.ones((2, ndims))
    mus[0, :] *= -1
    rads = np.ones(2) * 1.0
    covs = np.array([np.eye(ndims)] * 2)

    for alpha in [4, 5, 6]:
        k = '2,uniform,' + str(alpha)
        data[k], partsz[k] = generate_uniform(samplesz, ndims, mus, rads)
        data[k] = shift_clusters(data[k], partsz[k], alpha)

        k = '2,gaussian,' + str(alpha)
        data[k], partsz[k] = generate_multivariate_normal(samplesz, ndims, mus, covs)
        data[k] = shift_clusters(data[k], partsz[k], alpha)

        k = '2,quadratic,' + str(alpha)
        data[k], partsz[k] = generate_power(samplesz, ndims, 2, mus, rads)
        data[k] = shift_clusters(data[k], partsz[k], alpha)

        k = '2,exponential,' + str(alpha)
        data[k], partsz[k] = generate_exponential(samplesz, ndims, mus, rads)
        data[k] = shift_clusters(data[k], partsz[k], alpha)

    return data, partsz
    
##########################################################
def plot_points(data, outdir):
    info(inspect.stack()[0][3] + '()')
    figscale = 4
    fig, axs = plt.subplots(1, len(data.keys()), squeeze=False,
                figsize=(len(data.keys())*figscale, 1*figscale))
    for i, k in enumerate(data):
        max_ = np.max(np.abs(data[k])) * 1.15
        axs[0, i].scatter(data[k][:, 0], data[k][:, 1])
        axs[0, i].set_title(k)
        axs[0, i].set_xlim(-max_, +max_)
        axs[0, i].set_ylim(-max_, +max_)
    plt.savefig(pjoin(outdir, 'points_all.pdf'))

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
def plot_contour_power(ndims, power, mus, s, ax, cmap, linewidth, epsilon):
    X, Y, Z = mesh_xy(-1.0, +1.0, s)
    coords = np.zeros((s*s, 2), dtype=float)
    Zflat = np.zeros(s*s, dtype=float)
    # epsilon = .6

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coords[i*s + j, :] = np.array([X[i, j], Y[i, j]])

    for i in range(mus.shape[0]):
        mu = mus[i, :]
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
        mu = mus[i, :]
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
def get_descendants(z, nleaves, clustid, itself=True):
    """Get all the descendants from a given cluster id

    Args:
    z(np.ndarray): linkage matrix
    nleaves(int): number of leaves
    clustid(int): cluster id

    Returns:
    np.ndarray, np.ndarray: (leaves, links)
    """

    clustids = np.array([clustid]) if itself else np.array([])
    if clustid < nleaves: return clustids

    zid = int(clustid - nleaves)
    linkids1 = get_descendants(z, nleaves, z[zid, 0]) # left
    linkids2 = get_descendants(z, nleaves, z[zid, 1]) # right
    return np.concatenate((clustids, linkids1, linkids2)).astype(int)

##########################################################
def get_leaves(z, clustid):
    """Get leaves below clustid

    Args:
    z(np.ndarray): linkage matrix
    clustid(int): cluster id

    Returns:
    np.ndarray: leaves
    """
    # info(inspect.stack()[0][3] + '()')
    n = z.shape[0] + 1
    desc = get_descendants(z, n, clustid, itself=True)
    return desc[desc < n]

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
    links = get_descendants(linkageret, nleaves, parent)
    return (child in links)
    # if child in links: return True
    # else: return False

##########################################################
def get_parent(child, linkageret):
    zind = np.where(linkageret[:, 0:2] == child)[0]
    if len(zind) == 0:
        return None
    else:
        return zind[0] + len(linkageret) +1

##########################################################
def get_ancestors(child, linkageret):
    ancestors = []
    while True:
        ancestor = get_parent(child, linkageret)
        if ancestor == None: break
        ancestors.append(ancestor)
        child = ancestor
    return ancestors

##########################################################
def get_outermost_points(linkageret, outliersratio):
    if outliersratio > 0:
        outliers = []
        tree = scipy.cluster.hierarchy.to_tree(linkageret)
        n = tree.count
        m = tree.dist
        noutliers = outliersratio * n
        for i in range(n-1):
            if tree.left and tree.left.count <= noutliers:
                if tree.left:
                    leaves = get_leaves(linkageret, tree.left.id)
                    outliers.extend(leaves)
                tree = tree.right
            elif tree.right and tree.right.count <= noutliers:
                if tree.right:
                    leaves = get_leaves(linkageret, tree.right.id)
                    outliers.extend(leaves)
                tree = tree.left
            else:
                break
        return outliers, tree.dist
    else:
        return [], linkageret[-1, 2]

##########################################################
def find_clusters(data, linkageret, clsize, minnclusters, outliersratio):
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
    outliers, L = get_outermost_points(linkageret, outliersratio)

    counts = linkageret[:, 3]

    clustids = []
    for clustcount in range(clsize, n): # Find the clustids
        if len(clustids) >= minnclusters: break
        joininds = np.where(linkageret[:, 3] == clustcount)[0]

        for joinidx in joininds:
            clid = joinidx + n
            newclust = True
            for other in clustids:
                if is_child(clid, other, linkageret):
                    newclust = False
                    break
            if newclust:
                clustids.append(clid)

    if len(clustids) == 1:
        l = linkageret[clustids[0] - n, 2]
        return clustids, l, L, outliers
        

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
    avgheight = 0
    for cl in clustids:
        avgheight += linkageret[cl - n, 2]
    avgheight /= len(clustids) # average of the heights

    clustids = sorted(clustids)[:2]
    return clustids, avgheight, L, outliers

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
def export_individual_axis(ax, fig, labels, outdir, pad=0.3, prefix='', fmt='pdf'):
    n = ax.shape[0]*ax.shape[1]
    for k in range(n):
        i = k // ax.shape[1]
        j = k  % ax.shape[1]
        coordsys = fig.dpi_scale_trans.inverted()
        extent = ax[i, j].get_window_extent().transformed(coordsys)
        fig.savefig(pjoin(outdir, prefix + labels[k] + '.' + fmt),
                      bbox_inches=extent.expanded(1+pad, 1+pad))

##########################################################
def plot_contours(labels, outdir, icons=False):
    ndims = 2
    s = 500
    nrows = 2
    ncols = 4
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
    covs = np.array([np.eye(ndims) * 0.15]) # 1 gaussian
    plot_contour_gaussian(ndims, mus, covs, s, ax[0, 1], cmap, linewidth)

    plot_contour_power(ndims, 2, mus, s, ax[0, 2], cmap, linewidth, 1.4) # 1 quadratic

    mus = np.zeros((1, ndims))
    plot_contour_exponential(ndims, mus, s, ax[0, 3], cmap, linewidth) # 1 exponential

    c = 0.4
    mus = np.ones((2, ndims))*c; mus[1, :] *= -1
    rs = np.ones(2) * .4
    plot_contour_uniform(mus, rs, s, ax[1, 0], cmap, linewidth) # 2 uniform


    covs = np.array([np.eye(ndims) * 0.1] * 2)
    plot_contour_gaussian(ndims, mus, covs, s, ax[1, 1], cmap, linewidth) # 2 gaussians

    plot_contour_power(ndims, 2, mus, s, ax[1, 2], cmap, linewidth, .5) # 2 quadratic

    # mus = np.zeros((1, ndims))
    plot_contour_exponential(ndims, mus, s, ax[1, 3], cmap, linewidth) # 2 exponential


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
                # ax[i, j].set_xticks([])

    if icons:
        export_individual_axis(ax, fig, labels, outdir, 0, 'icon_', 'png')
    else:
        export_individual_axis(ax, fig, labels, outdir, .3, 'contour_')

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
def calculate_relevance(avgheight, maxdist):
    if avgheight > maxdist:
        return avgheight
    else:
        return (maxdist - avgheight) / maxdist

##########################################################
def compute_max_precision(clustids, partsz, z):
    precs = []
    for i in range(len(clustids)):
        rrobin = clustids[i:] + clustids[0:i]
        precs.append(compute_precision(rrobin, partsz, z))
    return np.max(precs)

##########################################################
def compute_precision(clustids, partsz, z):
    if len(clustids) != len(partsz): return 0

    tps = []
    fps = []
    idx = 0

    for i, cl in enumerate(clustids):
        leaves = get_leaves(z, cl)
        nextidx = idx + partsz[i]
        tps.append(np.sum( (leaves >= idx) & (leaves < nextidx)))
        fps.append(len(leaves) - tps[-1])
        idx = nextidx
    
    return np.sum(tps) / (np.sum(tps) + np.sum(fps))

##########################################################
def find_clusters_batch(data, metric, linkagemeths, clrelsize, nrealizations,
        outliersratio, palettehex, outdir):

    info('Computing relevances...')
    minnclusters = 2
    samplesz = data[list(data.keys())[0]].shape[0]
    ndims = data[list(data.keys())[0]].shape[1]
    clsize = int(clrelsize * samplesz)

    gtruths = compute_gtruth_vectors(data, nrealizations)
    info('Nrealizations:{}, Samplesize:{}, min nclusters:{}, min clustsize:{}'.\
         format(nrealizations, samplesz, minnclusters, clsize))

    rels = {}
    methprec = {}
    for k in data.keys():
        rels[k] = {l: [[], []] for l in linkagemeths}
        methprec[k] = {l: [] for l in linkagemeths}

    for r in range(nrealizations): # Compute relevances
        info('realization {:02d}'.format(r))
        data, partsz = generate_data(samplesz, ndims)

        for j, linkagemeth in enumerate(linkagemeths):
            for i, distrib in enumerate(data):
                try:
                    z = linkage(data[distrib], linkagemeth, metric)
                except Exception as e:
                    print(data[distrib], linkagemeth, metric)
                    np.save('/tmp/foo.npy', data[distrib])
                    print(e)
                    raise(e)

                ret = find_clusters(data[distrib], z, clsize,
                        minnclusters, outliersratio)
                clustids, avgheight, maxdist, outliers = ret
                rel = calculate_relevance(avgheight, maxdist)
                # TODO: compute precision here
                # prec = compute_precision(clustids, partsz[distrib], z)
                # methprec[distrib][linkagemeth].append(tp / (tp + fp))

                clustids = np.array(clustids)
                incinds = clustids - samplesz
                rels[distrib][linkagemeth][len(incinds)-1].append(rel)

    accrel = {k: {} for k in data.keys()} # accumulated relevances

    for i, distrib in enumerate(data):
        for linkagemeth in linkagemeths:
            accrel[distrib][linkagemeth] = np.zeros(2)
            for j, rel in enumerate(rels[distrib][linkagemeth]):
                accrel[distrib][linkagemeth][j] = np.sum(rel)

    diff = {} # difference to the ground-truth
    diffnorms = {}
    for k in data.keys():
        diff[k] = dict((el, np.zeros(2)) for el in linkagemeths)
        diffnorms[k] = {}

    for i, distrib in enumerate(data):
        for j, linkagemeth in enumerate(linkagemeths):
            diff[distrib][linkagemeth] = gtruths[distrib] - accrel[distrib][linkagemeth]
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
    plot_vectors(rels, accrel, methprec, gtruths, palettehex, outdir)

##########################################################
def plot_vectors(rels, accrel, methprec, gtruths, palettehex, outdir):
    info(inspect.stack()[0][3] + '()')
    distribs = list(rels.keys())
    linkagemeths = list(rels[distribs[0]].keys())
    nrealizations = np.sum([len(g) for g in rels[distribs[0]][linkagemeths[0]]])

    nrows = len(distribs); ncols = 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4), squeeze=False)
    palette = hex2rgb(palettehex, alpha=.8)

    origin = np.zeros(2)
    for i, distrib in enumerate(distribs):
        xs = np.array([gtruths[distrib][0]])
        ys = np.array([gtruths[distrib][1]])

        ax[i, 0].quiver(origin, origin, xs, ys, color='#000000', width=.01,
                        angles='xy', scale_units='xy', scale=1, label='Gtruth',
                        headwidth=5, headlength=4, headaxislength=3.5, zorder=3)

        for j, linkagemeth in enumerate(linkagemeths):
            xs = np.array([accrel[distrib][linkagemeth][0]])
            ys = np.array([accrel[distrib][linkagemeth][1]])

            coords = np.array([accrel[distrib][linkagemeth][0],
                              accrel[distrib][linkagemeth][1]])
            ax[i, 0].quiver(origin, origin, xs, ys, color=palette[j], width=.01,
                            angles='xy', scale_units='xy', scale=1,
                            label=linkagemeth,
                            headwidth=5, headlength=4, headaxislength=3.5,
                            zorder=1/np.linalg.norm(coords)+3)

            ax[i, 0].set_xlim(0, nrealizations)
            ax[i, 0].set_ylim(0, nrealizations)
            # ax[i, 0].set_axisbelow(True)
            # ax[i, 0].text(0.5, 0.9, 'prec:{}'.format(
                # np.sum(methprec[distrib][linkagemeth])),
                     # horizontalalignment='center', verticalalignment='center',
                     # fontsize='large')

        # plt.text(0.5, 0.9, 'winner:{}'.format(winner[distrib]),
                 # horizontalalignment='center', verticalalignment='center',
                 # fontsize='large', transform = ax[i, 1].transAxes)

        ax[i, 0].set_ylabel('Sum of relevances of 2 clusters', fontsize='medium')
        ax[i, 0].set_xlabel('Sum of relevances of 1 cluster', fontsize='medium')
        ax[i, 0].legend()

    plt.tight_layout(pad=4)
    export_individual_axis(ax, fig, distribs, outdir, 0.36, 'relev_vector_')

    for i, distrib in enumerate(distribs): # Plot
        ax[i, 0].set_ylabel('{}'.format(distrib), size='x-large')

    plt.savefig(pjoin(outdir, 'relev_vectors_all.pdf'))

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
def generate_dendrograms_all(data, metric, linkagemeths, clrelsize, pruningparam,
        palette, outdir):
    info(inspect.stack()[0][3] + '()')
    minnclusters = 2
    samplesz = data[list(data.keys())[0]].shape[0]
    ndims = data[list(data.keys())[0]].shape[1]
    clsize = int(clrelsize * samplesz)
    nlinkagemeths = len(linkagemeths)
    ndistribs = len(data.keys())
    nrows = ndistribs
    ncols = nlinkagemeths + 1
    figscale=5
    fig = plt.figure(figsize=(ncols*figscale, nrows*figscale))
    ax = np.array([[None]*ncols]*nrows)

    fig.suptitle('Sample size:{}, minnclusters:{},\nmin clustsize:{}'.\
                 format(samplesz, minnclusters, clsize), fontsize=24)

    nsubplots = nrows * ncols

    for subplotidx in range(nsubplots):
        i = subplotidx // ncols
        j = subplotidx % ncols

        if ndims == 3 and j == 0: proj = '3d'
        else: proj = None

        ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1, projection=proj)

    for i, k in enumerate(data):
        nclusters = int(k.split(',')[0])
        # plot_scatter(data[k], ax[i, 0], palette[1])
        ax[i, 0].scatter(data[k][:, 0], data[k][:, 1], c=palette[1])

        for j, l in enumerate(linkagemeths):
            z = linkage(data[k], l, metric)
            clustids, avgheight, maxdist, outliers = find_clusters(data[k], z,
                    clsize, minnclusters, pruningparam)
            rel = calculate_relevance(avgheight, maxdist)
            # plot_dendrogram(z, l, ax[i, j+1], rel, clustids, palette)
            plot_dendrogram(z, l, ax[i, j+1], 0, 0, clustids, ['k']*10, outliers)

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
def plot_article_uniform_distribs_scale(palette, outdir):
    info(inspect.stack()[0][3] + '()')
    samplesz = 250
    ndims = 2

    mus = np.ones((2, ndims))*.7
    mus[1, :] *= -1
    rs = np.ones(2) * .9
    coords2 = np.ndarray((0, 2))
    coords1, _ = generate_uniform(samplesz, ndims, mus, rs)
    # coords2 = generate_uniform(20, ndims, np.array([[.6, .4], [1, .6]]),
                               # np.ones(2) * .2)
    coords = np.concatenate((coords1, coords2))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
    # plot_scatter(coords, ax[0, 0], palette[1])
    ax[0, 0].scatter(coords[:, 0], coords[:, 1], c=palette[1])

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
    ax[0, 0].scatter([0], [0], c=palette[1]) # noise points
    ax[0, 0].set_xticks([-2.0, -1.0, 0, +1.0, +2.0])
    ax[0, 0].set_yticks([-2.0, -1.0, 0, +1.0, +2.0])

    export_individual_axis(ax, fig, ['2,uniform,rad0.9'], outdir, 0.3, 'points_')

##########################################################
def plot_article_gaussian_distribs_scale(palette, outdir):
    info(inspect.stack()[0][3] + '()')
    samplesz = 600
    ndims = 2

    mus = np.ones((2, ndims))*.7
    mus[1, :] *= -1
    covs = np.array([np.eye(2)*.15] * 2)
    coords2 = np.ndarray((0, 2))
    # coords1, _ = generate_gaussian(samplesz, ndims, mus, rs)
    coords1, _ = generate_multivariate_normal(samplesz, ndims, mus, covs)
    # coords2 = generate_uniform(20, ndims, np.array([[.6, .4], [1, .6]]),
                               # np.ones(2) * .2)
    coords = np.concatenate((coords1, coords2))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
    # plot_scatter(coords, ax[0, 0], palette[1])
    ax[0, 0].scatter(coords[:, 0], coords[:, 1], c=palette[1])

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
    ax[0, 0].scatter([0], [0], c=palette[1]) # noise points
    ax[0, 0].set_xticks([-2.0, -1.0, 0, +1.0, +2.0])
    ax[0, 0].set_yticks([-2.0, -1.0, 0, +1.0, +2.0])

    export_individual_axis(ax, fig, ['2,gaussian,0.15'], outdir, 0.3, 'points_')

##########################################################
def plot_dendrogram_clusters(data, partsz, validkeys, linkagemeth, metric, clrelsize,
        pruningparam, palettehex, outdir):

    info(inspect.stack()[0][3] + '()')
    minnclusters = 2
    nrows = len(validkeys)
    ndistribs = nrows
    ncols = 2
    samplesz = data[list(data.keys())[0]].shape[0]
    ndims = data[list(data.keys())[0]].shape[1]
    clsize = int(clrelsize * samplesz)

    nsubplots = nrows * ncols
    figscale = 5

    if ndims == 3:
        fig = plt.figure(figsize=(ncols*figscale, nrows*figscale))
        ax = np.array([[None]*ncols]*nrows)

        for subplotidx in range(nsubplots):
            i = subplotidx // ncols
            j = subplotidx % ncols

            if j == 0:
                ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1, projection='3d')
            else:
                ax[i, j] = fig.add_subplot(nrows, ncols, subplotidx+1)
    else:
        fig, ax = plt.subplots(nrows, ncols,
                figsize=(ncols*figscale, nrows*figscale), squeeze=False)

    for i, k in enumerate(validkeys):
        nclusters = int(k.split(',')[0])

        z = linkage(data[k], linkagemeth, metric)

        clustids, avgheight, maxdist, outliers = find_clusters(data[k], z, clsize,
                minnclusters, pruningparam)

        rel = calculate_relevance(avgheight, maxdist)
        prec = compute_max_precision(clustids, partsz[k], z)
        colours = plot_dendrogram(z, linkagemeth, ax[i, 1], avgheight,
                maxdist, clustids, palettehex, outliers)
        ax[i, 0].scatter(data[k][:, 0], data[k][:, 1], c=colours)
        xylim = np.max(np.abs(data[k])) * 1.1
        ax[i, 0].set_xlim(-xylim, +xylim)
        ax[i, 0].set_ylim(-xylim, +xylim)
        ax[i, 0].text(0.7, 0.9, '{:.2f}'.format(prec),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20)

        if len(clustids) == 1:
            text = 'rel: ({:.3f}, 0)'.format(rel)
        else:
            text = 'rel: (0, {:.3f})'.format(rel)
        # plt.text(0.7, 0.9, text,
                 # horizontalalignment='center', verticalalignment='center',
                 # fontsize=20, transform = ax[i, 1].transAxes)

    for i, k in enumerate(validkeys):
        ax[i, 0].set_ylabel(k, rotation=90, size=24)

    # fig.suptitle('Sample size:{}, minnclusters:{},\nmin clustsize:{}'.\
                 # format(samplesz, minnclusters, clsize),
                 # y=.92, fontsize=32)

    # plt.tight_layout(pad=1, h_pad=2, w_pad=3)
    plt.savefig(pjoin(outdir, '{}d-{}.pdf'.format(ndims, linkagemeth)))

    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])
            ax[i, j].tick_params(axis=u'both', which=u'both',length=0)
            ax[i, j].set_ylabel('')


    labels = []
    for v in validkeys:
        labels.append('hieclust_' + v + '_points')
        labels.append('hieclust_' + v + '_dendr')
    export_individual_axis(ax, fig, labels, outdir, pad=0.3, prefix='', fmt='pdf')

##########################################################
def plot_article_quiver(palettehex, outdir):
    info(inspect.stack()[0][3] + '()')
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
    ax.tick_params(axis='y', which='major', labelsize=25)
    ax.set_xticklabels([])
    ax.set_xlim(-.5, 7.5)
    ax.set_ylabel('Accumulated error', fontsize=25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.legend(
        fontsize=25,
        loc=[.82, .28],
    )

    ax.tick_params(bottom="off")
    ax.tick_params(axis='x', length=0)

    # axicon = fig.add_axes([0.4,0.4,0.1,0.1])
    # axicon.imshow(np.random.rand(100,100))
    # axicon.set_xticks([])
    # axicon.set_yticks([])

    trans = blended_transform_factory(fig.transFigure, ax.transAxes) # separator
    line = Line2D([0, .98], [-.05, -.05], color='k', transform=trans)
    plt.tight_layout()
    fig.lines.append(line)

##########################################################
def include_icons(iconpaths, fig):
    for i, iconpath in enumerate(iconpaths):
        # sign = 0.015*(-1) ** i
        sign = 0.0
        im = imageio.imread(iconpath)
        newax = fig.add_axes([0.17+i*.106, 0.79+sign, 0.06, 0.2], anchor='NE', zorder=-1)
        newax.imshow(im, aspect='equal')
        newax.axis('off')

##########################################################
def plot_parallel_all(df, outdir):
    info(inspect.stack()[0][3] + '()')
    if not os.path.isdir(outdir): os.mkdir(outdir)

    colours = cm.get_cmap('tab10')(np.linspace(0, 1, 6))
    dims = np.unique(df.dim)

    figscale = 5
    fig, axs = plt.subplots(len(dims), 1, figsize=(4*figscale, 5*figscale),
                            squeeze=False)

    for i, dim in enumerate(dims):
        slice = df[df.dim == dim]
        slice = slice.set_index('distrib')
        plot_parallel(slice, colours, axs[i, 0], fig)

    # plt.tight_layout(rect=(0.1, 0, 1, 1))
    plt.tight_layout(rect=(0.1, 0, 1, .94), h_pad=.6)
    for i, dim in enumerate(dims):
        plt.text(-0.12, .5, '{}-D'.format(dim),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize='30',
                 transform = axs[i, 0].transAxes
                 )

    iconpaths = [ pjoin(outdir, 'icon_' + f + '.png') for f in df[df.dim==2].distrib ]

    include_icons(iconpaths, fig)

    plt.savefig(pjoin(outdir, 'parallel_all.pdf'))

##########################################################
def count_method_ranking(df, linkagemeths, linkagemeth, outdir):
    info(inspect.stack()[0][3] + '()')
    
    methidx = np.where(np.array(linkagemeths) == linkagemeth)[0][0]

    data = []
    for i, row in df.iterrows():
        values = row[linkagemeths].values
        sortedidx = np.argsort(values)
        methrank = np.where(sortedidx == methidx)[0][0]

        data.append([row.distrib, row.dim, methrank])

    methdf = pd.DataFrame(data, columns='distrib,dim,methrank'.split(','))
    ndistribs = len(np.unique(methdf.distrib))

    fh = open(pjoin(outdir, 'meths_ranking.csv'), 'w')
    print('dim,uni,bi', file=fh)
    for d in np.unique(methdf.dim):
        filtered1 = methdf[methdf.dim == d][:4]
        filtered1 = filtered1[(filtered1.methrank < 3)]

        filtered2 = methdf[methdf.dim == d][4:]
        filtered2 = filtered2[(filtered2.methrank < 3)]

        m = methdf[methdf.dim == d]

        # print('dim:{}\t{}/{}\t{}/{}'.format(d, filtered1.shape[0], 4,
                                     # filtered2.shape[0], 4))
        print('{},{},{}'.format(d, filtered1.shape[0], filtered2.shape[0]), file=fh)
    fh.close()

##########################################################
def scatter_pairwise(df, linkagemeths, palettehex, outdir):
    info(inspect.stack()[0][3] + '()')

    nmeths = len(linkagemeths)
    nplots = int(scipy.special.comb(nmeths, 2))

    nrows = nplots;  ncols = 1
    figscale = 4
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(ncols*figscale*1.2, nrows*figscale))

    dims = np.unique(df.dim)
    distribs = []
    modal = {1: [], 2:[]}
    for d in np.unique(df.distrib):
        modal[int(d[0])].append(d)

    colours = {c: palettehex[i] for i,c in enumerate(dims)}
    markers = {1:'$1$', 2: '$2$'}

    corr =  np.ones((nmeths, nmeths), dtype=float)
    k = 0
    for i in range(nmeths-1):
        m1 = linkagemeths[i]
        for j in range(i+1, nmeths):
            ax = axs[k, 0]
            m2 = linkagemeths[j]

            for idx, row in df.iterrows():
                dim = row.dim
                nclusters = int(str(row.distrib)[0])
                ax.scatter(row[m1], row[m2], label=str(dim),
                           c=colours[dim], marker=markers[nclusters])

            p = pearsonr(df[m1], df[m2])[0]
            corr[i, j] = p
            corr[j, i] = p

            ax.set_title('Pearson corr: {:.3f}'.format(p))

            from matplotlib.patches import Patch
            legend_elements = [   Patch(
                                       facecolor=palettehex[dimidx],
                                       edgecolor=palettehex[dimidx],
                                       label=str(dims[dimidx]),
                                       )
                               for dimidx in range(len(dims))]

            # Create the figure
            ax.legend(handles=legend_elements, loc='lower right')
            # breakpoint()
            
            # ax.legend(title='Dimension', loc='lower right')
            ax.set_xlabel(m1)
            ax.set_ylabel(m2)
            ax.set_ylabel(m2)
            k += 1

    plt.tight_layout(pad=1, h_pad=3)
    plt.savefig(pjoin(outdir, 'meths_pairwise.pdf'))
    return corr

##########################################################
def plot_meths_heatmap(methscorr, linkagemeths, label, outdir):
    info(inspect.stack()[0][3] + '()')

    n = methscorr.shape[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(methscorr, cmap='coolwarm', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(linkagemeths)))
    ax.set_yticks(np.arange(len(linkagemeths)))
    ax.set_xticklabels(linkagemeths)
    ax.set_yticklabels(linkagemeths)
    ax.grid(False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(n):
        for j in range(i+1, n):
            text = ax.text(j, i, '{:.2f}'.format(methscorr[i, j]),
                           ha="center", va="center", color="k")


    # ax.set_title("Pairwise pearson correlation between linkage methods")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('pearson corr.', rotation=-90, va="bottom")
    fig.tight_layout(pad=0.5)
    plt.savefig(pjoin(outdir, 'meths_heatmap_' + label + '.pdf'))

##########################################################
def plot_graph(methscorr_in, linkagemeths, palettehex, label, outdir):
    """Plot the graph according to the weights.
    However, this is an ill posed problem because
    the weights would need to satisfy the triangle
    inequalities to allow this.

    Args:
    methscorr(np.ndarray): weight matrix
    linkagemeths(list of str): linkage methods
    outdir(str): output dir
    """
    info(inspect.stack()[0][3] + '()')

    methscorr = np.abs(methscorr_in)
    n = methscorr.shape[0]
    g = igraph.Graph.Full(n, directed=False, loops=False)

    min_, max_ = np.min(methscorr), np.max(methscorr)
    range_ = max_ - min_

    todelete = []
    widths = []
    for i in range(g.ecount()):
        e = g.es[i]
        c = methscorr[e.source, e.target]
        v = (c - min_) / range_
        if v > .3:
            g.es[i]['weight'] = (c - min_) / range_
            widths.append(10*g.es[i]['weight'])
        else:
            todelete.append(i)
    g.delete_edges(todelete)

    g.vs['label'] = linkagemeths
    # g.vs['label'] = ['     ' + l for l in linkagemeths]
    edgelabels = ['{:.2f}'.format(x) for x in g.es['weight']]
    # l = igraph.GraphBase.layout_fruchterman_reingold(weights=g.es['weight'])
    palette = hex2rgb(palettehex, alpha=.8)
    l = g.layout('fr', weights=g.es['weight'])
    outpath = pjoin(outdir, 'meths_graph_' + label + '.pdf')
    vcolors = palettehex[:g.vcount()]

    igraph.plot(g, outpath, layout=l,
                edge_label=edgelabels, edge_width=widths,
                edge_label_size=15,
                vertex_color=vcolors, vertex_frame_width=0,
                vertex_label_size=30,
                margin=80)

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ndims', type=int, default=2,
                        help='Dimensionality of the space')
    parser.add_argument('--samplesz', type=int, default=100, help='Sample size')
    parser.add_argument('--nrealizations', type=int, default=400, help='Sample size')
    parser.add_argument('--resultspath', default='/tmp/results.csv',
                        help='all results in csv format')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.INFO)

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(args.seed)

    linkagemeths = ['single', 'complete', 'average', 'centroid', 'median', 'ward']
    metric = 'euclidean'
    pruningparam = 0.02
    clrelsize = 0.3 # cluster rel. size
    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    info('pruningparam:{}'.format(pruningparam))

    validkeys = [
        '1,uniform',
        '1,gaussian',
        '1,quadratic',
        '1,exponential',
        '2,uniform,4',
        '2,gaussian,4',
        '2,quadratic,4',
        '2,exponential,4',
    ]

    data, partsz = generate_data(args.samplesz, args.ndims)
    # plot_points(data, args.outdir)
    # generate_dendrograms_all(data, metric, linkagemeths, clrelsize, pruningparam,
            # palettehex, args.outdir)
    plot_dendrogram_clusters(data, partsz, validkeys, 'single', metric, clrelsize,
            pruningparam, palettehex, args.outdir)
    # find_clusters_batch(data, metric, linkagemeths, clrelsize, args.nrealizations,
            # pruningparam, palettehex, args.outdir)
    # plot_contours(validkeys, args.outdir)
    # plot_contours(validkeys, args.outdir, True)
    # plot_article_uniform_distribs_scale(palettehex, args.outdir)
    # plot_article_gaussian_distribs_scale(palettehex, args.outdir)
    # plot_article_quiver(palettehex, args.outdir)
    
    # df = pd.read_csv(args.resultspath, sep='|')
    # df = df[df.distrib.isin(validkeys)]

    # plot_parallel_all(df, args.outdir)
    # count_method_ranking(df, linkagemeths, 'single', args.outdir)
    # for nclusters in ['1', '2']:
        # filtered = df[df['distrib'].str.startswith(nclusters)]
        # methscorr = scatter_pairwise(filtered, linkagemeths, palettehex, args.outdir)
        # plot_meths_heatmap(methscorr, linkagemeths, nclusters, args.outdir)
        # plot_graph(methscorr, linkagemeths, palettehex, nclusters, args.outdir)
    # test_inconsistency()

##########################################################
if __name__ == "__main__":
    main()

