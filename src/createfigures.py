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
    clcolours = [palette[0], palette[5]]
    outlierscolour = palette[1]

    for i, clustid in enumerate(clustids):
        links = utils.get_descendants(z, n, clustid)
        for l in links: colors[l]  = clcolours[i]

    ancestors = []
    for outlier in outliers:
        ancestors.extend(utils.get_ancestors(outlier, z))
    ancestors.extend(outliers)

    for anc in ancestors:
        # colors[anc]  = 'b'
        colors[anc]  = outlierscolour

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
        ax.axhline(y=avgheight/maxh, linestyle='--', linewidth=3, alpha=.7)
    if maxheight > 0:
        ax.axhline(y=maxheight/maxh, linestyle='--', c=outlierscolour,
                alpha=.7, linewidth=3)
    return colors[:n]

##########################################################
def plot_2coords(distribs, palettehex, outdir):
    info(inspect.stack()[0][3] + '()')
    data, partsz = utils.generate_data(distribs, 200, 2)
    figscale = 4
    fig, axs = plt.subplots(1, len(data.keys()), squeeze=False,
                figsize=(len(data.keys())*figscale, 1*figscale))
    for i, k in enumerate(data):
        max_ = np.max(np.abs(data[k])) * 1.15
        axs[0, i].scatter(data[k][:, 0], data[k][:, 1], c=palettehex[1])
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
    if len(mus) == 1:
        X, Y, Z = mesh_xy(0, +1.0, s)
    else:
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
def plot_contours(labels, outdir, icons=False):
    info(inspect.stack()[0][3] + '()')
    ndims = 2
    s = 500
    nrows = 2
    ncols = 4
    cmap = 'Blues'

    if icons:
        figscale = .5
        linewidth = 1
    else:
        figscale = 5
        linewidth = 3

    fig, ax = plt.subplots(nrows, ncols,
            figsize=(ncols*figscale, nrows*figscale), squeeze=False)

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
    if not icons: plt.savefig(pjoin(outdir, 'contour_all_{}d.pdf'.format(ndims)))

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
        utils.export_individual_axis(ax, fig, labels, outdir, 0, 'icon_', 'png')
    else:
        utils.export_individual_axis(ax, fig, labels, outdir, .3, 'contour_')

##########################################################
def plot_article_uniform_distribs_scale(palette, outdir):
    info(inspect.stack()[0][3] + '()')
    samplesz = 350
    ndims = 2

    mus = np.ones((2, ndims))*.7
    mus[1, :] *= -1
    r = 1.03
    rs = np.ones(2) * r
    coords2 = np.ndarray((0, 2))
    coords1, _ = utils.generate_uniform(samplesz, ndims, mus, rs)
    coords = np.concatenate((coords1, coords2))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
    ax[0, 0].scatter(coords[:, 0], coords[:, 1], c=palette[1])

    utils.export_individual_axis(ax, fig, ['2,uniform,rad{}'.format(r)],
            outdir, 0.3, 'points_')

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

    utils.export_individual_axis(ax, fig, ['2,gaussian,0.15'], outdir, 0.3, 'points_')

##########################################################
def plot_dendrogram_clusters(distribs, linkagemeths, metric, palettehex,
        ndims, outdir):
    info(inspect.stack()[0][3] + '()')
    # data, partsz = utils.generate_data(distribs, 200, ndims)
    distribs = '1 2 3'.split(' ')
    df = pd.read_csv('/home/frodo/results/graffiti/20200202-types/20200514-combine_me_he/clusters_eric_henrique.csv')
    df = pd.read_csv('/home/dufresne/temp/20200202-types/20200514-combine_me_he/clusters_eric_henrique.csv')
    data = {}
    partsz = {}
    for l in np.unique(df.label):
        filtered = df[df.label == l]
        lstr = str(l)
        data[lstr] = filtered[['x', 'y']].values
        partsz[lstr] = [filtered.shape[0]]
    clrelsize = .3
    minnclusters = 2
    nrows = len(distribs)
    ndistribs = nrows
    ncols = 3
    pruningparam = .02
    samplesz = data[list(data.keys())[0]].shape[0]
    # ndims = data[list(data.keys())[0]].shape[1]
    clsize = int(clrelsize * samplesz)

    nsubplots = nrows * ncols
    figscale = 5

    for linkagemeth in linkagemeths:
        fig, ax = plt.subplots(nrows, ncols,
                figsize=(ncols*figscale, nrows*figscale), squeeze=False)

        texts = []
        for i, k in enumerate(distribs):
            nclusters = int(k.split(',')[0])
            clsize = int(clrelsize * len(data[k]))

            z = linkage(data[k], linkagemeth, metric)
            maxdist = z[-1, 2]
            clustids, avgheight, outliersdist, outliers = utils.find_clusters(data[k], z,
                    clsize, minnclusters, pruningparam)

            rel = utils.calculate_relevance(avgheight, outliersdist, maxdist)
            prec = utils.compute_max_precision(clustids, partsz[k], z)
            colours = plot_dendrogram(z, linkagemeth, ax[i, 2], avgheight,
                    outliersdist, clustids, palettehex, outliers)
            clsizes = []
            for clid in clustids: clsizes.append(len(utils.get_leaves(z, clid)))
            texts.append(ax[i, 2].text(.8, .8, 'n:{}\nrel:{:.2f}\nprec:{:.2f}'.\
                    format(clsizes, rel, prec),
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=15, transform = ax[i, 2].transAxes))

            ax[i, 0].scatter(data[k][:, 0], data[k][:, 1], c=colours)
            ax[i, 0].set_title('Spatial coordinates (first cords)')
            xylim = np.max(np.abs(data[k][:, :2])) * 1.1
            ax[i, 0].set_xlim(-xylim, +xylim)
            ax[i, 0].set_ylim(-xylim, +xylim)
            ax[i, 0].set_ylabel(k, rotation=90, size=24)

            tr, evecs, evals = utils.pca(data[k], normalize=True)
            ax[i, 1].set_title('PCA components (pc1 and pc2)')
            ax[i, 1].scatter(tr[:, 0], tr[:, 1], c=colours)
            xylim = np.max(np.abs(tr[:, 0:2])) * 1.1
            ax[i, 1].set_xlim(-xylim, +xylim)
            ax[i, 1].set_ylim(-xylim, +xylim)


        plt.tight_layout(h_pad=3.5, w_pad=3.5)
        plt.savefig(pjoin(outdir, 'hieclust_{}_all.pdf'.format(linkagemeth)))

        for i in range(nrows):
            texts[i].set_visible(False)
            for j in range(2):
                ax[i, j].tick_params(axis=u'both', which=u'both',length=0)
                ax[i, j].set_ylabel('')

        labels = []
        for v in distribs:
            labels.append('hieclust_' + v + '_points_'+linkagemeth)
            labels.append('hieclust_' + v + '_pca_'+linkagemeth)
            labels.append('hieclust_' + v + '_dendr_'+linkagemeth)

        # utils.export_individual_axis(ax, fig, labels, outdir, pad=.5, fmt='pdf')

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
    ax.quiver(.53, 1.01, .1, .0, color=palettehex[0],
              width=.005, angles='xy', scale_units='xy', scale=1,
              headwidth=3, headlength=3, headaxislength=2, zorder=3)

    plt.text(0.45, 0.08, 'g',
             horizontalalignment='center', verticalalignment='center', style='italic',
             fontsize='x-large', transform = ax.transAxes)
    ax.quiver(1.55, .23, .1, .0, color='k',
              width=.005, angles='xy', scale_units='xy', scale=1,
              headwidth=3, headlength=3, headaxislength=2, zorder=3)

    plt.text(0.75, 0.3, 'l',
            horizontalalignment='center', verticalalignment='center', style='italic',
            color='#666666', fontname='serif',)

    ax.quiver(2.6, .69, .1, .0, color='#666666',
              width=.005, angles='xy', scale_units='xy', scale=1,
              headwidth=3, headlength=3, headaxislength=2, zorder=3)
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
    s = 4
    cols = list(dforig.columns)
    cols.remove('target')
    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    combs = list(itertools.combinations(cols, 2))
    clusters = np.unique(dforig['target'])

    for i, comb in enumerate(combs):
        try:
            fig, ax = plt.subplots(figsize=(s, .8*s))
            for cl in clusters:
                df = dforig[dforig['target'] == cl]
                ax.scatter(df[comb[0]], df[comb[1]], c=palettehex[1])
                ax.set_xlabel(comb[0])
                ax.set_ylabel(comb[1])
            ax.set_title('{}'.format(label))
            plt.tight_layout()
            plt.savefig(pjoin(outdir, '{}_{:03d}.pdf'.format(label, i)))
            plt.close()
        except Exception as e:
            info(e)

##########################################################
def plot_pca_first_coords(datasetsdir, outdir):
    s = 4
    def _plot_pca(dforig, label, outdir):
        df = dforig.select_dtypes(include=np.number)
        x = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].values
        transformed, eigvec, eigval = utils.pca(x)
        fig, ax = plt.subplots(figsize=(s, .8*s))
        palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.scatter(transformed[:, 0], transformed[:, 1], c=palettehex[1])
        ax.set_xlabel('PC0')
        ax.set_ylabel('PC1')
        ax.set_title('{} (PCA)'.format(label))
        plt.tight_layout()
        plt.savefig(pjoin(outdir, '{}_pca.pdf'.format(label)))
        plt.close()

    files = sorted(os.listdir(datasetsdir))
    for f in files: # cached datasets
        if not f.endswith('.csv'): continue
        label = f.replace('.csv', '')
        info('Plotting {}'.format(f))
        try:
            df = pd.read_csv(pjoin(datasetsdir, f))
            _plot_pca(df, label, outdir)
        except Exception as e:
            info(e)

    all = { #scikit datasets
            'iris': sklearn.datasets.load_iris(),
            'cancer': sklearn.datasets.load_breast_cancer(),
            'wine': sklearn.datasets.load_wine(),
            }

    for k, v in all.items():
        info('Plotting {}'.format(k))
        dforig = sklearn_to_df(v)
        _plot_pca(dforig, label, outdir)

##########################################################
def plot_real_datasets(datasetsdir, outdir):
    info(inspect.stack()[0][3] + '()')
    for f in sorted(os.listdir(datasetsdir)): # cached datasets
        if not f.endswith('.csv'): continue
        info('Plotting {}'.format(f))
        df = pd.read_csv(pjoin(datasetsdir, f))
        plot_combinations(df, f.replace('.csv', ''), outdir)

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
    parser.add_argument('--pardir', default='/tmp/out/', help='Parent dir')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.INFO)

    t0 = time.time()

    outdir = pjoin(args.pardir, 'figsarticle')
    if not os.path.isdir(outdir): os.mkdir(outdir)
    else: info('Overwriting contents of folder {}'.format(outdir))

    np.random.seed(args.seed)

    datasetsdir = './data/'
    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metric = 'euclidean'
    linkagemeths = 'single,complete,average,centroid,median,ward'.split(',')
    decays = 'uniform,gaussian,power,exponential'.split(',')

    alpha = '4'
    distribs = [','.join(['1', d]) for d in decays]
    distribs += [','.join(['2', d, alpha]) for d in decays]
    # distribs += [','.join(['2', d, '5']) for d in decays]
    # distribs += [','.join(['2', d, '6']) for d in decays]

    realdir = pjoin(outdir, 'realplots/')
    if not os.path.isdir(realdir): os.mkdir(realdir)
    # plot_real_datasets(datasetsdir, realdir)
    # plot_pca_first_coords(datasetsdir, realdir)
    # plot_2coords(distribs, palettehex, outdir)
    plot_dendrogram_clusters(distribs, linkagemeths, metric, palettehex,
            2, outdir)
    # plot_contours(distribs, outdir)
    # plot_contours(distribs, outdir, True)
    # plot_article_uniform_distribs_scale(palettehex, outdir)
    # plot_article_gaussian_distribs_scale(palettehex, outdir)
    # plot_article_quiver(palettehex, outdir)
    info('Elapsed time:{}'.format(time.time()-t0))
    info('Results are in {}'.format(outdir))
    
##########################################################
if __name__ == "__main__":
    main()

