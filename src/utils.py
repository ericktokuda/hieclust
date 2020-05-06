#!/usr/bin/env python3
"""Benchmark on hierarchical clustering methods
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
import os
import inspect
import numpy as np

import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt; plt.style.use('ggplot')

import scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn import preprocessing

##########################################################
def multivariate_normal(x, mean, cov):
    """P.d.f. of the multivariate normal when the covariance matrix is positive definite.
    Source: wikipedia"""
    ndims = len(mean)
    B = x - mean
    return (1. / (np.sqrt((2 * np.pi)**ndims * np.linalg.det(cov))) *
            np.exp(-0.5*(np.linalg.solve(cov, B).T.dot(B))))

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
def generate_power(samplesz, ndims, power, mus, rads, positive=False):
    ncenters = len(mus)
    x = np.ndarray((samplesz*ncenters, ndims), dtype=float)

    for i in range(ncenters):
        aux = rads[i] * (1 - np.random.power(a=power+1, size=(samplesz, ndims)))
        if positive and i % 2 ==1:
            aux = rads[i] - aux
        if not positive:
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
    ndims = x.shape[1]
    d1 = x[:partsz[0]]
    d2 = x[partsz[0]:]
    stdavg = np.mean([np.std(d1), np.std(d2)])

    d = alpha * stdavg / 2.0
    mu = d / np.sqrt(ndims)
    shifted = x.copy()

    shifted[:partsz[0]] += (+mu) - np.mean(d1)
    shifted[partsz[0]:] += (-mu) - np.mean(d2)
    # calculate_alpha(shifted, partsz)
    return shifted

##########################################################
def generate_data(distribs, samplesz, ndims):
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
    if k in distribs:
        data[k], partsz[k] = generate_uniform(samplesz, ndims, mu, r)
 
    k = '1,gaussian'
    cov = np.array([np.eye(ndims)])*.3
    if k in distribs:
        data[k], partsz[k] = generate_multivariate_normal(samplesz, ndims, mu, cov)

    k = '1,power'
    if k in distribs:
        data[k], partsz[k] = generate_power(samplesz, ndims, 2, mu, np.array([1])*1,
                positive=True)

    k = '1,exponential'
    if k in distribs:
        data[k], partsz[k] = generate_exponential(samplesz, ndims, mu, np.ones(1)*.3)

    mus = np.ones((2, ndims))
    mus[0, :] *= -1
    rads = np.ones(2) * 1.0
    covs = np.array([np.eye(ndims)] * 2)

    for alpha in [4,5,6]:
        k = '2,uniform,' + str(alpha)
        if k in distribs:
            data[k], partsz[k] = generate_uniform(samplesz, ndims, mus, rads)
            data[k] = shift_clusters(data[k], partsz[k], alpha)

        k = '2,gaussian,' + str(alpha)
        if k in distribs:
            data[k], partsz[k] = generate_multivariate_normal(samplesz, ndims, mus, covs)
            data[k] = shift_clusters(data[k], partsz[k], alpha)

        k = '2,power,' + str(alpha)
        if k in distribs:
            data[k], partsz[k] = generate_power(samplesz, ndims, 2, mus, rads,
                    positive=True)
            data[k] = shift_clusters(data[k], partsz[k], alpha)

        k = '2,exponential,' + str(alpha)
        if k in distribs:
            data[k], partsz[k] = generate_exponential(samplesz, ndims, mus, rads)
            data[k] = shift_clusters(data[k], partsz[k], alpha)

    return data, partsz
    
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
def get_outermost_points(linkageret, outliersratio, hfloor):
    # TODO: remove intersection of outliers and cluster links
    if outliersratio > 0:
        outliers = []
        tree = scipy.cluster.hierarchy.to_tree(linkageret)
        n = tree.count
        m = tree.dist
        noutliers = outliersratio * n
        for i in range(n-1):
            if tree.dist <= hfloor: break

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

    hfloor = np.max(clustids) - n # highest link
    hfloor = linkageret[hfloor, 2]

    outliers, L = get_outermost_points(linkageret, outliersratio, hfloor)
    # TODO: get closest number to the requested

    if len(clustids) == 1:
        l = linkageret[clustids[0] - n, 2]
        return np.array(clustids), l, L, outliers
        

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

    clustids = np.array(sorted(clustids)[:2])
    return clustids, avgheight, L, outliers

##########################################################
def export_individual_axis(ax, fig, labels, outdir, pad=0.3, prefix='', fmt='pdf'):
    n = ax.shape[0]*ax.shape[1]
    for k in range(n):
        i = k // ax.shape[1]
        j = k  % ax.shape[1]
        ax[i, j].set_title('')

    for k in range(n):
        i = k // ax.shape[1]
        j = k  % ax.shape[1]
        coordsys = fig.dpi_scale_trans.inverted()
        extent = ax[i, j].get_window_extent().transformed(coordsys)
        x0, y0, x1, y1 = extent.extents

        if isinstance(pad, list):
            x0 -= pad[0]; y0 -= pad[1]; x1 += pad[2]; y1 += pad[3];
        else:
            x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;

        bbox =  matplotlib.transforms.Bbox.from_extents(x0, y0, x1, y1)
        fig.savefig(pjoin(outdir, prefix + labels[k] + '.' + fmt),
                      bbox_inches=bbox)
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

    plot_contour_power(ndims, 2, mus, s, ax[0, 2], cmap, linewidth, 1.4,
            positive=True) # 1 quadratic

    mus = np.zeros((1, ndims)) # 1 exponential
    plot_contour_exponential(ndims, mus, s, ax[0, 3], cmap, linewidth)

    c = 0.4
    mus = np.ones((2, ndims))*c; mus[1, :] *= -1
    rs = np.ones(2) * .4
    plot_contour_uniform(mus, rs, s, ax[1, 0], cmap, linewidth) # 2 uniform


    covs = np.array([np.eye(ndims) * 0.1] * 2) # 2 gaussians
    plot_contour_gaussian(ndims, mus, covs, s, ax[1, 1], cmap, linewidth)

    plot_contour_power(ndims, 2, mus*.3, s, ax[1, 2], cmap, linewidth, .5,
            positive=True) # 2 quadratic

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
def calculate_relevance(avgheight, outliersdist, maxdist):
    if avgheight > outliersdist:
        return avgheight / maxdist
    else:
        return (outliersdist - avgheight) / maxdist

##########################################################
def compute_max_precision(clustids, partsz, z):
    precs = []
    clids = list(clustids)
    for i in range(len(clids)):
        rrobin = clids[i:] + clids[0:i]
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
def accumulate_relevances(rels, distribs, linkagemeths):
    accrel = {k: {} for k in distribs} # accumulated relevances

    for i, distrib in enumerate(distribs):
        for linkagemeth in linkagemeths:
            accrel[distrib][linkagemeth] = np.zeros(2)
            for j, rel in enumerate(rels[distrib][linkagemeth]):
                accrel[distrib][linkagemeth][j] = np.sum(rel)
    return accrel


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
def pca(xin, normalize=False):
    x = xin.copy()

    if normalize: x = preprocessing.normalize(x, axis=0)

    x -= np.mean(x, axis=0) # just centralize
    cov = np.cov(x, rowvar = False)
    evals , evecs = np.linalg.eig(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx] # each column is a eigenvector
    evals = evals[idx]
    a = np.dot(x, evecs)
    return a, evecs, evals

