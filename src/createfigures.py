#!/usr/bin/env python3
"""Benchmark on hierarchical clustering methods
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
import os
import time
import numpy as np
import scipy
import inspect

import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt; plt.style.use('ggplot')
import matplotlib.cm as cm
from matplotlib.patches import Circle
import itertools

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import pandas as pd
import sklearn.datasets

import utils

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
        links = utils.get_descendants(z, n, clustid)
        for l in links: colors[l]  = c

    c = vividcolors.pop()
    ancestors = []
    for outlier in outliers:
        ancestors.extend(utils.get_ancestors(outlier, z))
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
def plot_2coords(data, outdir):
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
    xs = np.linspace(min, max, s)
    ys = np.linspace(min, max, s)
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
def plot_contour_power(ndims, power, mus, s, ax, cmap, linewidth, epsilon,
        positive=False):
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

        if positive:
            outside = (coords < mu) if mu[0] >= 0 else (coords > mu)
            d[outside[:, 0] + outside[:, 1]] = 0
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
            Z[i, j] += utils.multivariate_normal(coords[cind, :], mus[mind, :],
                                           covs[mind, :, :])

    contours = ax.contour(X, Y, Z, cmap=cmap, levels=3, linewidths=linewidth)
    return ax

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

    plot_contour_power(ndims, 2, mus, s, ax[0, 2], cmap, linewidth, 1.4,
            positive=True) # 1 quadratic

    mus = np.zeros((1, ndims))
    plot_contour_exponential(ndims, mus, s, ax[0, 3], cmap, linewidth) # 1 exponential

    c = 0.4
    mus = np.ones((2, ndims))*c; mus[1, :] *= -1
    rs = np.ones(2) * .4
    plot_contour_uniform(mus, rs, s, ax[1, 0], cmap, linewidth) # 2 uniform


    covs = np.array([np.eye(ndims) * 0.1] * 2)
    plot_contour_gaussian(ndims, mus, covs, s, ax[1, 1], cmap, linewidth) # 2 gaussians

    plot_contour_power(ndims, 2, mus*.3, s, ax[1, 2], cmap, linewidth, .5,
            positive=True) # 2 quadratic

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
            clustids, avgheight, maxdist, outliers = utils.find_clusters(data[k], z,
                    clsize, minnclusters, pruningparam)
            rel = utils.calculate_relevance(avgheight, maxdist)
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
    coords1, _ = utils.generate_uniform(samplesz, ndims, mus, rs)
    # coords2 = utils.generate_uniform(20, ndims, np.array([[.6, .4], [1, .6]]),
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
    coords1, _ = utils.generate_multivariate_normal(samplesz, ndims, mus, covs)
    # coords2 = utils.generate_uniform(20, ndims, np.array([[.6, .4], [1, .6]]),
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
        clustids, avgheight, maxdist, outliers = utils.find_clusters(data[k], z, clsize,
                minnclusters, pruningparam)

        rel = utils.calculate_relevance(avgheight, maxdist)
        prec = utils.compute_max_precision(clustids, partsz[k], z)
        colours = plot_dendrogram(z, linkagemeth, ax[i, 1], avgheight,
                maxdist, clustids, palettehex, outliers)
        clsizes = []
        for clid in clustids: clsizes.append(len(utils.get_leaves(z, clid)))
        ax[i, 1].text(.8, .8, 'n:{}\nrel:{:.2f}\nprec:{:.2f}'.\
                format(clsizes, rel, prec),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=15, transform = ax[i, 1].transAxes)

        ax[i, 0].scatter(data[k][:, 0], data[k][:, 1], c=colours)
        xylim = np.max(np.abs(data[k])) * 1.1
        ax[i, 0].set_xlim(-xylim, +xylim)
        ax[i, 0].set_ylim(-xylim, +xylim)


    for i, k in enumerate(validkeys):
        ax[i, 0].set_ylabel(k, rotation=90, size=24)

    # fig.suptitle('Sample size:{}, minnclusters:{},\nmin clustsize:{}'.\
                 # format(samplesz, minnclusters, clsize),
                 # y=.92, fontsize=32)

    # plt.tight_layout(pad=1, h_pad=2, w_pad=3)
    plt.savefig(pjoin(outdir, 'hieclust_{}_all.pdf'.format(linkagemeth)))

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
              width=.010, angles='xy', scale_units='xy', scale=1,
              headwidth=4, headlength=4, headaxislength=3, zorder=3)

    ax.quiver(orig[0], orig[1]+delta, v2[0]-delta, v2[1],
              color='#000000', alpha=.7,
              width=.013, angles='xy', scale_units='xy', scale=1,
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
    ax.set_ylabel('Relevance of 2 clusters', fontsize='large')
    ax.set_xlabel('Relevance of 1 cluster', fontsize='large')
    ax.set_xticks([0, 1, 2, 3.0])
    ax.set_yticks([0, 1, 2])
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2)
    plt.tight_layout(pad=1)

    plt.savefig(pjoin(outdir, 'vector.pdf'))

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

##########################################################
def plot_combinations(dforig, label, outdir):
    s = 5
    cols = list(dforig.columns)
    cols.remove('target')
    combs = list(itertools.combinations(cols, 2))
    clusters = np.unique(dforig['target'])

    for i, comb in enumerate(combs):
        fig, ax = plt.subplots(figsize=(s, s))
        for cl in clusters:
            df = dforig[dforig['target'] == cl]
            ax.scatter(df[comb[0]], df[comb[1]], c='brown')
            ax.set_xlabel(comb[0])
            ax.set_ylabel(comb[1])
        plt.tight_layout(pad=3)
        plt.suptitle('{} dataset'.format(label))
        plt.savefig(pjoin(outdir, '{}_{:03d}.png'.format(label, i)))
        plt.close()

##########################################################
def plot_real_datasets(outdir):
    datasetsdir = 'data/'
    for f in os.listdir(datasetsdir): # cached datasets
        if not f.endswith('.csv'): continue
        info('Plotting {}'.format(f))
        df = pd.read_csv(pjoin(datasetsdir, f))
        plot_combinations(df, f.replace('.csv', ''), outdir)

    return
    all = { #scikit datasets
            'iris': sklearn.datasets.load_iris(),
            'cancer': sklearn.datasets.load_breast_cancer(),
            'wine': sklearn.datasets.load_wine(),
            }

    for k, v in all.items():
        info('Plotting {}'.format(k))
        dforig = sklearn_to_df(v)
        plot_combinations(dforig, k, outdir)

    return

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ndims', type=int, default=2, help='Dimensionality')
    parser.add_argument('--samplesz', type=int, default=200, help='Sample size')
    parser.add_argument('--pardir', default='/tmp/hieclust/', help='Parent dir')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.INFO)

    t0 = time.time()

    outdir = pjoin(args.pardir, 'figsarticle')
    if not os.path.isdir(outdir): os.mkdir(outdir)
    else: info('Overwriting contents of folder {}'.format(outdir))

    np.random.seed(args.seed)

    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metric = 'euclidean'

    linkagemeths = 'single,complete,average,centroid,median,ward'.split(',')
    decays = 'uniform,gaussian,power,exponential'.split(',')
    alpha = '4'

    distribs = [','.join(['1', d]) for d in decays]
    distribs += [','.join(['2', d, alpha]) for d in decays]
    metric = 'euclidean'
    pruningparam = 0.02
    clrelsize = 0.3 # cluster rel. size
    precthresh = 0.7

    realdir = pjoin(outdir, 'realplots/')
    if not os.path.isdir(realdir): os.mkdir(realdir)
    plot_real_datasets(realdir)
    return
    data, partsz = utils.generate_data(distribs, args.samplesz, args.ndims)
    plot_2coords(data, outdir)
    generate_dendrograms_all(data, metric, linkagemeths, clrelsize,
            pruningparam, palettehex, outdir)
    plot_dendrogram_clusters(data, partsz, distribs, 'single', metric,
            clrelsize, pruningparam, palettehex, outdir)
    plot_contours(distribs, outdir)
    plot_contours(distribs, outdir, True)
    plot_article_uniform_distribs_scale(palettehex, outdir)
    plot_article_gaussian_distribs_scale(palettehex, outdir)
    plot_article_quiver(palettehex, outdir)
    info('Elapsed time:{}'.format(time.time()-t0))
    info('Results are in {}'.format(outdir))
    
##########################################################
if __name__ == "__main__":
    main()

