#!/usr/bin/env python3
"""Generate article plots """

import argparse
from os.path import join as pjoin
from logging import debug, info
import os, sys, time, datetime, inspect, random
import numpy as np
import scipy

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
from myutils import create_readme
from myutils.plot import export_subplots

##########################################################
def plot_dendrogram(z, linkagemeth, ax, avgheight, clustids, palette, outliers):
    """Call fancy scipy.dendogram with @clustids colored and with a line with height
    given by @lthresh """

    dists = z[:, 2]
    # dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
    maxh = np.max(dists)
    dists = dists / maxh
    z[:, 2] = dists
    n = z.shape[0] + 1
    colors = (2 * n - 1) * ['k']
    clcolours = [palette[0], palette[5], palette[1], palette[2], palette[3]]
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
    # if maxheight > 0:
        # ax.axhline(y=maxheight/maxh, linestyle='--', c=outlierscolour,
                # alpha=.7, linewidth=3)
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
    plt.close()

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
def plot_contours(outdir, icons=False):
    info(inspect.stack()[0][3] + '()')
    alpha = '4'
    decays = 'uniform,gaussian,power,exponential'.split(',')
    distribs = [','.join(['1', d, '0']) for d in decays]
    distribs += [','.join(['2', d, alpha]) for d in decays]
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
        export_subplots(ax, fig, distribs, outdir, 0, 'icon_', 'png')
    else:
        export_subplots(ax, fig, distribs, outdir, .3, 'contour_')

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

    linkagemeth = 'average'
    plot_raw_dendrogram(coords, linkagemeth, 5,
                        pjoin(outdir, 'dendr2_{}.pdf'.format(linkagemeth)))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
    ax[0, 0].scatter(coords[:, 0], coords[:, 1], c=palette[1])

    export_subplots(ax, fig, ['2,uniform,rad{}'.format(r)],
            outdir, 0.3, 'points_')

##########################################################
def plot_article_uniform_distribs_scale3(palette, outdir):
    info(inspect.stack()[0][3] + '()')
    samplesz = 350
    ndims = 2

    # mus = np.ones((2, ndims))*.7
    # mus[1, :] *= -1
    mus = np.array([
        [np.cos(0), np.sin(0)],
        [np.cos(2 / 3 * np.pi), np.sin(2 / 3 * np.pi)],
        [np.cos(4 / 3 * np.pi), np.sin(4 / 3 * np.pi)],
    ])
    r = .80
    rs = np.ones(3) * r
    rs = [.7, .8, .8]
    coords2 = np.ndarray((0, 2))
    coords1, _ = utils.generate_uniform(samplesz, ndims, mus, rs)
    coords = np.concatenate((coords1, coords2))

    linkagemeth = 'average'
    plot_raw_dendrogram(coords, linkagemeth, 5,
                        pjoin(outdir, 'dendr3_{}.pdf'.format(linkagemeth)))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
    ax[0, 0].scatter(coords[:, 0], coords[:, 1], c=palette[1])

    plt.tight_layout()
    export_subplots(ax, fig, ['3,uniform,rad{}'.format(r)],
            outdir, 0.4, 'points_')

##########################################################
def plot_article_uniform_distribs_scale4(palette, outdir):
    info(inspect.stack()[0][3] + '()')
    samplesz = 350
    ndims = 2

    # mus = np.ones((2, ndims))*.7
    # mus[1, :] *= -1
    # mus = np.ones((4, ndims))*.7
    mus = np.array([
        [-.9, -.8],
        [-.8,  .9],
        [ .8,  .5],
        [ .9,  -.9],
    ])
    r = .80
    rs = np.ones(3) * r
    rs = [.6, .8, .7, .6]
    coords2 = np.ndarray((0, 2))
    coords1, _ = utils.generate_uniform(samplesz, ndims, mus, rs)
    coords = np.concatenate((coords1, coords2))

    linkagemeth = 'average'
    plot_raw_dendrogram(coords, linkagemeth, 5,
                        pjoin(outdir, 'dendr4_{}.pdf'.format(linkagemeth)))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
    ax[0, 0].scatter(coords[:, 0], coords[:, 1], c=palette[1])

    plt.tight_layout()
    export_subplots(ax, fig, ['4,uniform,rad{}'.format(r)],
            outdir, 0.4, 'points_')

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
def plot_dendrogram_clusters(k, distribs, linkagemeths, metric, palettehex,
                             ndims, outdir):
    info(inspect.stack()[0][3] + '()')
    samplesz = 200
    data, partsz = utils.generate_data(distribs, samplesz, ndims)
    clrelsize = .3
    clsize = int(clrelsize * samplesz)
    nrows = len(distribs)
    ndistribs = nrows
    ncols = 3
    pruningparam = .03

    nsubplots = nrows * ncols
    figscale = 4

    for linkagemeth in linkagemeths:
        fig, ax = plt.subplots(nrows, ncols,
                figsize=(1.2*ncols*figscale, nrows*figscale), squeeze=False)

        texts = []
        for i, distrib in enumerate(distribs):
            nclusters = int(distrib.split(',')[0])

            d = data[distrib]
            z, clustids, outliers = utils.find_clusters(d, k, linkagemeth,
                    metric, clsize, pruningparam)

            rel = utils.calculate_relevance(z, clustids)
            colours = plot_dendrogram(z, linkagemeth, ax[i, 2], rel,
                    clustids, palettehex, outliers)
            texts.append(ax[i, 2].text(.8, .8, 'rel:{:.2f}'.\
                    format(rel),
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=15, transform = ax[i, 2].transAxes))
            # texts.append(ax[i, 2].text(.8, .8, 'n:{}\nrel:{:.2f}\nprec:{:.2f}'.\
                    # format(clsizes, rel, prec),
                    # horizontalalignment='center', verticalalignment='center',
                    # fontsize=15, transform = ax[i, 2].transAxes))

            ax[i, 0].scatter(d[:, 0], d[:, 1], c=colours)
            ax[i, 0].set_title('Spatial coordinates (first cords)')
            xylim = np.max(np.abs(d[:, :2])) * 1.1
            ax[i, 0].set_xlim(-xylim, +xylim)
            ax[i, 0].set_ylim(-xylim, +xylim)

            fac = 1.5
            ax[i, 0].set_xlim(-fac*6, +fac*6)
            ax[i, 0].set_ylim(-6, +6)
            # ax[i, 0].set_aspect('equal', adjustable='box')

            ax[i, 0].set_ylabel(k, rotation=90, size=24)

            tr, evecs, evals = utils.pca(d, normalize=True)
            ax[i, 1].set_title('PCA components (pc1 and pc2)')
            ax[i, 1].scatter(tr[:, 0], tr[:, 1], c=colours)
            xylim = np.max(np.abs(tr[:, 0:2])) * 1.1
            ax[i, 1].set_xlim(-xylim, +xylim)
            ax[i, 1].set_ylim(-xylim, +xylim)

        plt.tight_layout(h_pad=3.5, w_pad=3.5)
        plt.savefig(pjoin(outdir, 'dendrclusters_{}.pdf'.format(linkagemeth)))
        plt.close()

##########################################################
def plot_dendrogram_clusters_overlap(linkagemeths, metric, palettehex, ndims, outdir):
    alphas =  np.arange(2, 7, .5) # overlap
    distribs = [ '2,overlap,{}'.format(a) for a in alphas ]
    k = 2
    plot_dendrogram_clusters(k, distribs, linkagemeths, metric, palettehex,
                             ndims, outdir)

##########################################################
def plot_dendrogram_clusters_inbalance(linkagemeths, metric, palettehex, ndims, outdir):
    ratios = np.arange(.500, .761, 0.02) # inbalance
    distribs = [ '2,inbalance,{:.02f}'.format(r) for r in ratios ]
    k = 2
    plot_dendrogram_clusters(k, distribs, linkagemeths, metric, palettehex,
                             ndims, outdir)
##########################################################
def plot_article_quiver(palettehex, outdir):
    info(inspect.stack()[0][3] + '()')
    fig, ax = plt.subplots(figsize=(4, 3.5))
    orig = np.zeros(2)
    v1 = np.array([0.6, 1.8])
    v2 = np.array([3.0, 0.0])
    v3 = v2 - v1
    delta = .01

    matplotlib.rcParams['text.usetex'] = True
    ax.quiver(orig[0], orig[1], v1[0], v1[1], color=palettehex[2],
              width=.010, angles='xy', scale_units='xy', scale=1,
              headwidth=4, headlength=4, headaxislength=3, zorder=3)
    ax.quiver(orig[0], orig[1]+delta, v2[0]-delta, v2[1],
              color=palettehex[1],
              width=.013, angles='xy', scale_units='xy', scale=1,
              headwidth=4, headlength=4, headaxislength=3, zorder=3)
    ax.quiver(v1[0] + delta, v1[1] - delta, v3[0]-delta, v3[1]+delta,
              color='#999999', alpha=1.0, linestyle='dashed',
              width=.007, angles='xy', scale_units='xy', scale=1,
              headwidth=5, headlength=8, headaxislength=5, zorder=3)

    ax.scatter([0.6], [1.8], s=2, c='k')

    plt.text(0.16, 0.47, r'\overrightarrow{r}',
             horizontalalignment='center', verticalalignment='center',
             color=palettehex[2],
             style='italic', fontsize='xx-large', transform = ax.transAxes)
    plt.text(0.45, 0.08, r'\overrightarrow{g}',
             color=palettehex[1],
             horizontalalignment='center', verticalalignment='center',
             style='italic', fontsize='xx-large', transform = ax.transAxes)
    plt.text(.63, 0.4, r'\overrightarrow{l}',
             horizontalalignment='center', verticalalignment='center',
             color='#666666', style='italic',
             fontsize='xx-large', transform = ax.transAxes)

    ax.set_ylabel(r'$R_2$', fontsize='large')
    ax.set_xlabel(r'$R_1$', fontsize='large')
    ax.set_xticks([0, 1, 2, 3.0])
    ax.set_yticks([0, 1, 2])
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2)
    plt.tight_layout(pad=1)
    plt.savefig(pjoin(outdir, 'vector.pdf'))
    plt.close()

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
            fig, ax = plt.subplots(figsize=(s, .75*s))
            for cl in clusters:
                df = dforig[dforig['target'] == cl]
                ax.scatter(df[comb[0]], df[comb[1]], c=palettehex[1], s=8,
                           alpha=.4, edgecolors='none')
                ax.set_xlabel(comb[0])
                ax.set_ylabel(comb[1])
            # ax.set_title('{}'.format(label))
            plt.tight_layout()
            plt.savefig(pjoin(outdir, '{}_{:03d}.png'.format(label, i)))
            plt.close()
        except Exception as e:
            info(e)

##########################################################
def plot_combinations3d(dforig, label, outdir):
    s = 4
    cols = list(dforig.columns)
    cols.remove('target')
    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    combs = list(itertools.combinations(cols, 3))
    clusters = np.unique(dforig['target'])

    for i, comb in enumerate(combs):
        try:
            # fig, ax = plt.subplots(figsize=(s, .8*s))
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for cl in clusters:
                df = dforig[dforig['target'] == cl]
                # print(comb, cl)
                ax.scatter(df[comb[0]], df[comb[1]], df[comb[2]], c=palettehex[1])
                ax.set_xlabel(comb[0])
                ax.set_ylabel(comb[1])
            ax.set_title('{}'.format(label))
            plt.tight_layout()
            plt.savefig(pjoin(outdir, '{}_{:03d}.png'.format(label, i)))
            plt.close()
        except Exception as e:
            info(e)

##########################################################
def plot_combinations_selected(df, label, outdir):

    s = 4
    cols = list(df.columns)
    cols.remove('target')
    plt.style.use('default')
    palettehex = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']
    # palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    combs = list(itertools.combinations(cols, 3))
    clusters = np.unique(df['target'])

    sel = dict(
        abalone = [['Shell weight', 'Shucked weight']],
        bank_marketing = [['duration', 'balance']],
        diabetes = [['BMI', 'Glucose']],
        cancer = [['mean symmetry', 'mean smoothness'],
                  ['worst concave points', 'mean concave points'],
                  ['texture error', 'mean symmetry']],
        heart = [['chol', 'age']],
        sonar = [['freq54', 'freq00'],
                 ['freq49', 'freq06']],
    )

    for i, comb in enumerate(combs):

        match = False
        for feats in sel[label]:
            if feats[0] in comb and feats[1] in comb:
                match = True
                break
        if not match: continue

        try:
            for j, alpha in enumerate(np.arange(0, 360, 2)):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(projection='3d')

                ax.view_init(elev=10., azim=alpha)
                ax.scatter(df[comb[0]], df[comb[1]], df[comb[2]], c=palettehex[1])
                ax.set_xlabel(comb[0])
                ax.set_ylabel(comb[1])
                ax.set_zlabel(comb[2])

                # ax.set_title('{}'.format(label))
                plt.tight_layout()
                plt.savefig(pjoin(outdir, '{}_{:03d}_{:02d}.png'.format(label, i, j)))
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
def plot_real_datasets_selected(datasetsdir, outdir):
    info(inspect.stack()[0][3] + '()')
    for f in sorted(os.listdir(datasetsdir)): # cached datasets
        if not f.endswith('.csv'): continue
        name = f.split('.csv')[0]
        if not name in ['abalone', 'diabetes', 'heart', 'sonar', 'bank_marketing']:
            continue
        info('Plotting {}'.format(f))
        df = pd.read_csv(pjoin(datasetsdir, f))
        plot_combinations_selected(df, f.replace('.csv', ''), outdir)

    all = { #scikit datasets
            'cancer': sklearn.datasets.load_breast_cancer(),
            }

    for k, v in all.items():
        info('Plotting {}'.format(k))
        dforig = sklearn_to_df(v)
        plot_combinations_selected(dforig, k, outdir)

    return

##########################################################
def plot_real_datasets3d(datasetsdir, outdir):
    info(inspect.stack()[0][3] + '()')
    for f in sorted(os.listdir(datasetsdir)): # cached datasets
        if not f.endswith('.csv'): continue
        info('Plotting {}'.format(f))
        df = pd.read_csv(pjoin(datasetsdir, f))
        plot_combinations3d(df, f.replace('.csv', ''), outdir)

    all = { #scikit datasets
            'iris': sklearn.datasets.load_iris(),
            'cancer': sklearn.datasets.load_breast_cancer(),
            'wine': sklearn.datasets.load_wine(),
            }

    for k, v in all.items():
        info('Plotting {}'.format(k))
        dforig = sklearn_to_df(v)
        plot_combinations3d(dforig, k, outdir)

    return

##########################################################
def plot_raw_dendrogram(data, linkagemeth, figscale, outpath):
    """Plot non-fancy dendrogram."""
    info(inspect.stack()[0][3] + '()')
    z = linkage(data, 'average')
    figscale = 5
    fig, ax = plt.subplots(1, figsize=(1.5*figscale, figscale))
    dendrogram(
        z,
        color_threshold=0,
        truncate_mode=None,
        show_contracted=False,
        show_leaf_counts=False,
        no_labels=True,
        ax=ax,
        link_color_func=lambda k: 'k',
    )
    plt.savefig(outpath)
    plt.close()

##########################################################
def plot_dummy_dendrogram(outdir):
    plt.style.use('ggplot')
    np.random.seed(0)
    eps = .05
    x = np.array([
        [1,1],
        [2,1],
        [4,1],
        [1,10],
        [1.5,10],
        [4,10],
        ])
    Z = linkage(x, 'ward')
    print(Z)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].scatter(x[:, 0], x[:, 1])
    for i, coords in enumerate(x):
        ax[0].annotate(str(i), (coords[0]+eps, coords[1]+eps))

    ax[0].set_xlim(0.5, 4.5)
    ax[0].set_ylim(0, 12)

    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
        ax=ax[1]
    )
    ax[1].set_ylim(0, 17)
    plt.tight_layout(pad=1, w_pad=2)
    plt.savefig(pjoin(outdir, 'dendr.png'))

##########################################################
def main(outdir):
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(0); random.seed(0)

    datasetdir = './data/'
    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metric = 'euclidean'
    linkagemeths = ['single', 'complete', 'average', 'centroid', 'median', 'ward']
    decays = ['uniform', 'gaussian', 'power', 'exponential']

    # alpha = '4'
    # distribs = [','.join(['1', d, '0']) for d in decays]
    # distribs += [','.join(['2', d, alpha]) for d in decays]
    # distribs = ['1,gaussian,0', '2,gaussian,4', '3,gaussian,4', '4,gaussian,4']
    # distribs += [','.join(['2', d, '5']) for d in decays]
    # distribs += [','.join(['2', d, '6']) for d in decays]

    # plot_real_datasets(datasetdir, outdir)
    # plot_real_datasets_selected(datasetdir, outdir)
    # plot_dendrogram_clusters_inbalance(linkagemeths, metric, palettehex, 2, outdir)
    # plot_dendrogram_clusters_overlap(linkagemeths, metric, palettehex, 2, outdir)
    # plot_real_datasets3d(datasetdir, outdir)
    # plot_pca_first_coords(datasetdir, outdir)
    # plot_contours(outdir, icons=False)
    # plot_contours(outdir, icons=True)
    # plot_article_uniform_distribs_scale(palettehex, outdir)
    # plot_article_uniform_distribs_scale3(palettehex, outdir)
    # plot_article_uniform_distribs_scale4(palettehex, outdir)
    plot_article_gaussian_distribs_scale(palettehex, outdir)
    # plot_article_quiver(palettehex, outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Results are in {}'.format(outdir))

##########################################################
if __name__ == "__main__":

    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

