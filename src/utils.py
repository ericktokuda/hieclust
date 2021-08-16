#!/usr/bin/env python3
"""Benchmark on hierarchical clustering methods
"""

import argparse
from os.path import join as pjoin
import os
import inspect
from math import acos
import numpy as np

import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt; plt.style.use('ggplot')

import scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
from scipy.optimize import brentq
import pandas as pd
from sklearn import preprocessing
from myutils import info

##########################################################
NOTFOUND = -1

##########################################################
def multivariate_normal(x, mean, cov):
    """P.d.f. of the multivariate normal when the covariance matrix is positive
    definite. Source: wikipedia"""
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
def generate_multivariate_normal(samplesz, ndims, ncenters, covs=[]):
    x = np.ndarray((samplesz*ncenters, ndims), dtype=float)
    mu0 = np.zeros(ndims)

    for i in range(ncenters):
        cov = covs[i, :, :]
        aux = np.random.multivariate_normal(mu0, cov, size=samplesz)
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
def shift_clusters2(x, partsz, alpha):
    """Shift 2 clusters(k=2) accordign to
    alpha = 2*d / std
    where d is the distance between the cluster and the origin
    Resulting clusters are symmetric wrt the origin """

    ndims = x.shape[1]
    d1 = x[:partsz[0]]
    d2 = x[partsz[0]:]
    stdavg = np.mean([np.std(d1), np.std(d2)])

    d = alpha * stdavg / 2.0
    mu = d / np.sqrt(ndims)
    shifted = x.copy()

    shifted[:partsz[0]] += (+mu) - np.mean(d1)
    shifted[partsz[0]:] += (-mu) - np.mean(d2)
    return shifted

##########################################################
def shift_clusters3(data, partsz, alpha, stdavg):
    """Shift 3 clusters (k=3)"""

    l = alpha * stdavg
    sq3 = np.sqrt(3) # Coords from an equilateral triangle
    mus3 = np.array([[0, l * sq3 / 4],
                     [l / 2, -l * sq3 / 4],
                     [-l / 2, -l * sq3 / 4]])
    shifted = data.copy()
    inds = np.cumsum(partsz)
    inds = np.insert(inds, 0, 0)
    for i in range(len(inds) -1):
        shifted[inds[i]:inds[i+1]] +=  mus3[i, :]
    return shifted

##########################################################
def shift_clusters4(data, partsz, alpha, stdavg):
    """Shift 4 clusters (k=4)"""

    l = alpha * stdavg
    v = l / 2
    mus4 = np.array([ [v, v], [v, -v], [-v, -v], [-v, v],])
    shifted = data.copy()

    inds = np.cumsum(partsz)
    inds = np.insert(inds, 0, 0)
    for i in range(len(inds) -1):
        shifted[inds[i]:inds[i+1]] +=  mus4[i, :]

    return shifted

##########################################################
def generate_data_k1(data, partsz, samplesz, ndims, distribs):
    """Generate data for k=1"""
    mu = np.zeros((1, ndims))

    b = '1,uniform'
    r = np.array([1])
    if b in distribs:
        data[b], partsz[b] = generate_uniform(samplesz, ndims, mu, r)

    b = '1,gaussian'
    cov = np.array([np.eye(ndims)])*.3
    if b in distribs:
        data[b], partsz[b] = generate_multivariate_normal(samplesz, ndims, 1, cov)

    b = '1,power'
    if b in distribs:
        data[b], partsz[b] = generate_power(samplesz, ndims, 2, mu, np.array([1])*1,
                positive=True)

    b = '1,exponential'
    if b in distribs:
        data[b], partsz[b] = generate_exponential(samplesz, ndims, mu, np.ones(1)*.3)

    return data, partsz

##########################################################
def generate_data_k2(data, partsz, samplesz, ndims, mus2, covs2, distribs):
    """Generate data for k=2"""
    rads = np.ones(2) * 1.0

    for alpha in [4,5,6]:
        b = '2,uniform,' + str(alpha)
        if b in distribs:
            data[b], partsz[b] = generate_uniform(samplesz, ndims, mus2, rads)
            data[b] = shift_clusters2(data[b], partsz[b], alpha)

        b = '2,gaussian,' + str(alpha)
        if b in distribs:
            data[b], partsz[b] = generate_multivariate_normal(samplesz, ndims, 2,
                                                              covs2)
            data[b] = shift_clusters2(data[b], partsz[b], alpha)

        b = '2,power,' + str(alpha)
        if b in distribs:
            data[b], partsz[b] = generate_power(samplesz, ndims, 2, mus2, rads,
                    positive=True)
            data[b] = shift_clusters2(data[b], partsz[b], alpha)

        b = '2,exponential,' + str(alpha)
        if b in distribs:
            data[b], partsz[b] = generate_exponential(samplesz, ndims, mus2, rads)
            data[b] = shift_clusters2(data[b], partsz[b], alpha)
    return data, partsz

##########################################################
def generate_data_hdbscan(data, partsz, samplesz, ndims, distribs):
    """Generate HDBSCAN data"""
    if '5,hdbscan' in distribs:
        data['5,hdbscan'] = np.load(open('/tmp/clusterable_data.npy', 'rb'))
    return data, partsz


##########################################################
def generate_data_overlap(data, partsz, samplesz, ndims, mus2, covs2, distribs):
    """Generate data with overlap for k=2"""
    for b in distribs: # Data with overlap
        if not 'overlap' in b: continue
        alpha = float(b.split(',')[-1])
        data[b], partsz[b] = generate_multivariate_normal(samplesz, ndims, 2, covs2)
        data[b] = shift_clusters2(data[b], partsz[b], alpha)
    return data, partsz

##########################################################
def generate_data_inbalance(data, partsz, samplesz, ndims, mus2, covs2, distribs):
    """Generate data with inbalance for k=2"""
    alpha = 5 # a bit more separated
    for b in distribs: # Data with imbalance
        if not 'imbalance' in b: continue
        ratio = float(b.split(',')[-1])
        sz1 = int(ratio * samplesz); sz2 = samplesz - sz1
        mask = np.zeros(2*samplesz, dtype=bool); mask[:sz1] = 1
        ldata, lpartsz = generate_multivariate_normal(2*samplesz, ndims, 2, covs2)
        mask[lpartsz[0]:lpartsz[0]+sz2] = 1
        partsz[b] = np.array([sz1, sz2])
        data[b] = shift_clusters2(ldata[mask, :], partsz[b], alpha)
    return data, partsz

##########################################################
def generate_data_k3(data, partsz, samplesz, ndims, distribs):
    """Generate data for k=3"""
    covs3 = np.array([np.eye(ndims)] * 3)
    stdavg = 1
    alpha = 4

    b = '3,gaussian,{}'.format(alpha)
    if not (b in distribs): return data, partsz

    data[b], partsz[b] = generate_multivariate_normal(samplesz, ndims, 3, covs3)
    data[b] = shift_clusters3(data[b], partsz[b], alpha, stdavg)
    return data, partsz

##########################################################
def generate_data_k4(data, partsz, samplesz, ndims, distribs):
    """Generate data for k=3"""
    covs4 = np.array([np.eye(ndims)] * 4)
    stdavg = 1
    alpha = 4

    b = '4,gaussian,{}'.format(alpha)
    if not (b in distribs): return data, partsz

    data[b], partsz[b] = generate_multivariate_normal(samplesz, ndims, 4, covs4)
    data[b] = shift_clusters4(data[b], partsz[b], alpha, stdavg)
    return data, partsz

##########################################################
def generate_data(distribs, samplesz, ndims):
    """Synthetic data """
    # info(inspect.stack()[0][3] + '()')

    data = {}
    partsz = {}

    mus2 = np.ones((2, ndims)); mus2[0, :] *= -1
    covs2 = np.array([np.eye(ndims)] * 2)

    data, partsz = generate_data_k1(data, partsz, samplesz, ndims, distribs)
    data, partsz = generate_data_k2(data, partsz, samplesz, ndims,
                                    mus2, covs2, distribs)
    data, partsz = generate_data_hdbscan(data, partsz, samplesz, ndims, distribs)
    data, partsz = generate_data_overlap(data, partsz, samplesz, ndims,
                                         mus2, covs2, distribs)
    data, partsz = generate_data_inbalance(data, partsz, samplesz, ndims,
                                           mus2, covs2, distribs)
    data, partsz = generate_data_k3(data, partsz, samplesz, ndims, distribs)
    data, partsz = generate_data_k4(data, partsz, samplesz, ndims, distribs)

    return data, partsz

##########################################################
def plot_data(data, partsz, outdir):
    """Plot sets of points generated by generate_data."""
    info(inspect.stack()[0][3] + '()')
    axlimfactor = 3
    alpha = .6

    for b in data.keys():
        ngroups = int(b.split(',')[0])
        axlim = axlimfactor * ngroups
        a = data[b]; bb = partsz[b]
        plt.close();
        idx = 0
        W = 500; H = 500
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        for i in range(len(bb)):
            ax.scatter(a[idx:idx+bb[i], 0], a[idx:idx+bb[i], 1],
                        alpha=alpha, label='Group{}'.format(i), linewidths=0,
                       # c=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], 
                       s=10)
            idx += bb[i]

        ax.set_xlim(-axlim, +axlim); plt.ylim(-axlim, +axlim)
        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.tight_layout()
        # plt.legend()
        plt.savefig(pjoin(outdir, '{}.png'.format(b)))

##########################################################
def get_descendants(z, nleaves, clustid, itself=True):
    """Get all the descendants from a given cluster id"""

    clustids = np.array([clustid]) if itself else np.array([])
    if clustid < nleaves: return clustids

    zid = int(clustid - nleaves)
    linkids1 = get_descendants(z, nleaves, z[zid, 0]) # left
    linkids2 = get_descendants(z, nleaves, z[zid, 1]) # right
    return np.concatenate((clustids, linkids1, linkids2)).astype(int)

##########################################################
def get_leaves(z, clustid):
    """Get leaves below clustid """
    n = len(z) + 1
    desc = get_descendants(z, n, clustid, itself=True)
    return desc[desc < n]

##########################################################
def get_nleaves(z, clustid):
    """Get leaves below clustid """
    return len(get_leaves(z, clustid))

##########################################################
def get_leaves_all(z, clustids):
    """Get leaves below clustid """
    leaves = []
    for clid in clustids:
        leaves.extend(get_leaves(z, clid))
    return leaves

##########################################################
def is_child(parent, child, linkageret):
    """Check if @child is a direct child of @parent """

    nleaves = linkageret.shape[0] + 1
    links = get_descendants(linkageret, nleaves, parent)
    return (child in links)
    # if child in links: return True
    # else: return False

##########################################################
def get_parent(linkageret, child):
    zind = np.where(linkageret[:, 0:2] == child)[0]
    if len(zind) == 0:
        return None
    else:
        return zind[0] + len(linkageret) + 1

##########################################################
def get_ancestors(child, linkageret):
    ancestors = []
    while True:
        ancestor = get_parent(linkageret, child)
        if ancestor == None: break
        ancestors.append(ancestor)
        child = ancestor
    return ancestors

##########################################################
def identify_outliers(linkageret, outliersratio, hfloor):
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
def get_cluster(clsize, clustids, z):
    """Try to find a cluster of minimum size @clsize in the data given by @z"""
    n = len(z) + 1
    allleaves = set(list(range(n)))
    visitted = set(get_leaves_all(z, clustids))
    nonvisitted = allleaves.difference(visitted)

    if clsize > (len(allleaves) - len(visitted)): return NOTFOUND
    newclustids = []

    for leaf in nonvisitted:
        u = leaf
        while(get_nleaves(z, u) < clsize):
            visitted = set(get_leaves_all(z, clustids))
            uleaves = set(get_leaves(z, u))
            if len(uleaves.intersection(visitted)) > 0: break
            u = get_parent(z, u)

        leavesu = set(get_leaves(z, u))
        inters = leavesu.intersection(visitted)
        if len(inters) > 0: continue # If any leaf had already been found, abort
        if len(leavesu) == clsize: return u # The new cluster size is exactly clsize
        newclustids.append(u)

    if len(newclustids) == 0: return NOTFOUND

    minid = -1; mindiff = n

    for clid in newclustids:
        curdiff = np.abs(get_nleaves(z, clid) - clsize)
        if curdiff < mindiff:
            minid = clid; mindiff = curdiff

    return minid

##########################################################
def get_last_valid_link(clid, clsize, z):
    """ Get the cluster with size closest to the requested among clid
    and its children"""
    n = len(z) + 1
    diff0 = np.abs(get_nleaves(z, clid) - clsize)
    child1, child2 = int(z[clid -n, 0]), int(z[clid -n, 1])
    diff1 = np.abs(get_nleaves(z, child1) - clsize)
    diff2 = np.abs(get_nleaves(z, child2) - clsize)
    ind = np.argmin([diff0, diff1, diff2])
    if ind == 0: return clid
    elif ind == 1: return child1
    elif ind == 2: return child2

##########################################################
def get_highest_cluster(z, clustids):
    """Get the id of the cluster in @clustids with highest height."""
    n = len(z) + 1
    maxid = -1
    maxh = -1
    for clid in clustids:
        h = z[clid - n, 2]
        if h > maxh: maxid = clid
    return maxid

##########################################################
def find_clusters(data, k, linkagemeth, metric, clsize, outliersratio):
    """Return npred, rel, feats, prec"""
    z = linkage(data, linkagemeth, metric)
    n = len(z) + 1

    for i in range(k, 0, -1): # Find the max number of clusters up to k
        clustids = []
        found = False

        while(len(clustids)) < i:
            u = get_cluster(clsize, clustids, z)
            if u == NOTFOUND: break
            v = get_last_valid_link(u, clsize, z)
            clustids.append(v)

        if len(clustids) == i:
            found = True
            break

    if not found: return [], [], [] # In case no cluster is found

    hfloor = get_highest_cluster(z, clustids) # The maximum the outliers can descend
    outliers, L = identify_outliers(z, outliersratio, hfloor)

    return z, clustids, outliers

##########################################################
def export_individual_axis(ax, fig, labels, outdir, pad=0.3, prefix='', fmt='pdf'):
    n = ax.shape[0]*ax.shape[1]
    for b in range(n):
        i = k // ax.shape[1]
        j = k  % ax.shape[1]
        ax[i, j].set_title('')

    for b in range(n):
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
def hex2rgb(hexcolours, normalized=False, alpha=None):
    rgbcolours = np.zeros((len(hexcolours), 3), dtype=int)
    for i, h in enumerate(hexcolours):
        rgbcolours[i, :] = np.array([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])

    if alpha != None:
        aux = np.zeros((len(hexcolours), 4), dtype=float)
        aux[:, :3] = rgbcolours / 255
        aux[:, -1] = .7 # alpha
        rgbcolours = aux
    elif normalized:
        rgbcolours = rgbcolours.astype(float) / 255

    return rgbcolours

##########################################################
def calculate_relevance(z, clustids):
    maxdist = z[-1, 2]
    n = len(z) + 1
    acc = 0
    for cl in clustids:
        acc += z[cl - n, 2]
    return acc / len(clustids) / maxdist

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
    accrel = {k: {} for b in distribs} # accumulated relevances

    for i, distrib in enumerate(distribs):
        for linkagemeth in linkagemeths:
            accrel[distrib][linkagemeth] = np.zeros(2)
            for j, rel in enumerate(rels[distrib][linkagemeth]):
                accrel[distrib][linkagemeth][j] = np.sum(rel)
    return accrel

##########################################################
def compute_gtruth_vectors(maxnclu, distribs, nrealizations):
    """Compute the ground-truth as proposed by Luciano """
    gtruths = np.zeros((len(distribs), maxnclu), dtype=float)
    for i, d in enumerate(distribs):
        n = int(d.split(',')[0])
        gtruths[i, n - 1] = nrealizations # n-1 because Python is 0-index based
    return gtruths

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
