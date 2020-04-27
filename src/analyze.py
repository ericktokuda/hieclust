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

import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt; plt.style.use('ggplot')
import matplotlib.cm as cm
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D

from scipy.stats import pearsonr
from scipy import stats
import pandas as pd

import imageio
import scipy
import inspect
import igraph

import utils

##########################################################
def concat_results(resdir):
    info(inspect.stack()[0][3] + '()')
    filenames = ['results.csv', 'resultsall.csv']

    for f in filenames:
        dfpath = pjoin(resdir, f)
        if os.path.exists(dfpath):
            info('Loading {}'.format(dfpath))
            return pd.read_csv(dfpath, sep='|')

    csvs = []
    for d in os.listdir(resdir):
        respath = pjoin(resdir, d, 'results.csv')
        if not os.path.exists(respath): continue
        csvs.append(pd.read_csv(respath, sep='|'))
    resdf = pd.concat(csvs, axis=0, ignore_index=True)
    resdf.to_csv(dfpath, sep='|', index=False)
    return resdf

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
    # ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.tick_params(axis='y', which='major', labelsize=25)
    ax.set_xticklabels([])
    ax.set_xlim(-.5, 7.5)
    ax.set_ylabel('Accum. difference', fontsize=25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.legend(
        fontsize=25,
        loc=[.82, .28],
        # loc=[.88, .18],
    )

    ax.tick_params(bottom="off")
    ax.tick_params(axis='x', length=0)

    # axicon = fig.add_axes([0.4,0.4,0.1,0.1])
    # axicon.imshow(np.random.rand(100,100))
    # axicon.set_xticks([])
    # axicon.set_yticks([])

    trans = blended_transform_factory(fig.transFigure, ax.transAxes) # separator
    line = Line2D([0.05, .99], [-.02, -.02], color='k', transform=trans)
    plt.tight_layout()
    fig.lines.append(line)

##########################################################
def include_icons(iconpaths, fig):
    for i, iconpath in enumerate(iconpaths):
        # sign = 0.015*(-1) ** i
        sign = 0.0
        im = imageio.imread(iconpath)
        newax = fig.add_axes([0.185+i*.103, 0.79+sign, 0.06, 0.2], anchor='NE', zorder=-1)
        newax.imshow(im, aspect='equal')
        newax.axis('off')

##########################################################
def plot_parallel_all(df, iconsdir, outdir):
    info(inspect.stack()[0][3] + '()')
    if not os.path.isdir(iconsdir):
        info('Icons path {} does not exist,'.format(iconsdir));
        info('run src/createfigures.py first!');
        return
    if not os.path.isdir(outdir): os.mkdir(outdir)

    colours = cm.get_cmap('tab10')(np.linspace(0, 1, 6))
    dims = np.unique(df.dim)

    figscale = 5
    # fig, axs = plt.subplots(len(dims), 1, figsize=(6*figscale, 8*figscale),
    fig, axs = plt.subplots(len(dims), 1, figsize=(4*figscale, 5*figscale),
                            squeeze=False)

    for i, dim in enumerate(dims):
        slice = df[df.dim == dim]
        slice = slice.set_index('distrib')
        plot_parallel(slice, colours, axs[i, 0], fig)

    # plt.tight_layout(rect=(0.1, 0, 1, 1))
    plt.tight_layout(rect=(0.08, 0, 1, .94), h_pad=.3)
    for i, dim in enumerate(dims):
        # plt.text(-0.1, .5, '{}-D'.format(dim),
        plt.text(-0.15, .5, '{}-D'.format(dim),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize='30',
                 transform = axs[i, 0].transAxes
                 )

    iconpaths = [ pjoin(iconsdir, 'icon_' + f + '.png') for f in df[df.dim==2].distrib ]

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
    palette = utils.hex2rgb(palettehex, alpha=.8)
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
def find_diff_neigh(j, hs, dx, dir):
    jj = j + dir
    while True:
        if dx[jj] != 0: break
        jj = jj + dir
    return jj, hs[jj]

##########################################################
def update_zero_derivative_points(x):
    dh = np.diff(x, axis=1, prepend=-0.01) 
    zeroinds = np.where(dh == 0)

    for i in range(len(zeroinds[0])):
        coord = [zeroinds[0][i], zeroinds[1][i]]
        lind, lval = find_diff_neigh(coord[1], x[coord[0], :],
                dh[coord[0], :], -1)
        rind, rval = find_diff_neigh(coord[1], x[coord[0], :],
                dh[coord[0], :], +1)
        newv = lval + ( (coord[0] - lind) / (rind - lind) ) * (rval - lval)
        x[coord[0], coord[1]] = newv
    return x

##########################################################
def plot_pca(dforig, cols, colstointerpolate, palettehex, label, ax):
    info(inspect.stack()[0][3] + '()')
    df = dforig.copy()

    if len(colstointerpolate) > 0:
        xx = df[colstointerpolate].values
        df[colstointerpolate] = update_zero_derivative_points(xx)

    x = df[cols].values
    transformed, evecs, evals = utils.pca(x, normalize=True)
    # print(label, len(cols), transformed.shape)

    contribs = []

    for i in [0, 1]:
        evec = np.abs(evecs[:, i])
        contrib = evec / np.sum(evec)
        ids = np.argsort(contrib)[-1:]
        contrib = contrib[ids]
        contribcols = np.array(cols)[ids]
        contribs.append('{} ({:.03f})'.format(contribcols[0], contrib[0]))

    t = 0
    for i, d in enumerate(np.unique(df.distrib)):
        idx = np.where(df.distrib == d)[0]
        # print(np.linalg.norm(transformed[idx, :] - t))
        t = transformed[idx, :]
        ax.scatter(t[:, 0], t[:, 1], label=d, c=palettehex[i], alpha=.7, s=4)

    # ax.text(.1, .8, 'PC0:{}\nPC1:{}'.format(contribs[0], contribs[1]),
            # transform=ax.transAxes)
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')
    ax.set_title('PCA - {}'.format(label))
    ax.legend(fancybox=True, framealpha=0.6, markerscale=2)
    return transformed

##########################################################
def filters_by_dim(resdf, dims):
    info(inspect.stack()[0][3] + '()')
    resdf = resdf[resdf.dim.isin(dims)]
    return resdf

##########################################################
def plot_link_heights(dforig, hcols, palettehex, ax):
    info(inspect.stack()[0][3] + '()')
    for i, d in enumerate(np.unique(dforig.distrib)):
        df = dforig[dforig.distrib == d]
        y = df.mean()[hcols]
        ax.scatter(range(len(y)), y, c=palettehex[i], label=d, s=4)
    ax.set_xlabel('Linkage id')
    ax.set_ylabel('Relative height')
    ax.set_title('Average rel. height')
    ax.legend()

##########################################################
def plot_attrib(df, attrib, palettehex, ax):
    nbins = 6
    try:
        plot_attrib_density(df, attrib, palettehex, ax)
    except:
        ax.clear()
        plot_attrib_hist(df, attrib, palettehex, ax)
    ax.set_title('{} distribution'.format(attrib))

##########################################################
def plot_attrib_hist(dforig, attrib, palettehex, ax):
    nbins = 6
    for i, d in enumerate(np.unique(dforig.distrib)):
        df = dforig[dforig.distrib == d]
        y = df[attrib]
        ax.hist(y, nbins, label=d, histtype='bar',
                color=palettehex[i], alpha=0.7)
    ax.set_xlabel(attrib)
    ax.legend()

##########################################################
def plot_attrib_density(dforig, attrib, palettehex, ax):
    for i, d in enumerate(np.unique(dforig.distrib)):
        df = dforig[dforig.distrib == d]
        y = df[attrib]
        kde = stats.gaussian_kde(y)
        xx = np.linspace(np.min(y), np.max(y), 100)
        ax.plot(xx, kde(xx), c=palettehex[i], label=d)
    ax.set_xlabel(attrib)
    ax.legend()

##########################################################
def plot_fitted_heights(dforig, nheights, coeffcols, palettehex, ax):
    # from numpy.polynomial.polynomial import polyval
    xs = list(range(nheights))
    for i, d in enumerate(np.unique(dforig.distrib)):
        df = dforig[dforig.distrib == d]
        coeffs = np.mean(df[coeffcols], axis=0) # highest degree coeff first
        ys = np.polyval(coeffs, xs)
        ax.plot(xs, ys, c=palettehex[i], label=d)
        ax.set_xlabel('Relative height')
        ax.set_xlabel('Link id')
        ax.set_title('HeightsFit (poly3)')
        ax.legend()

##########################################################
def analyze_features(featpath, label, palettehex, outdir):
    info(inspect.stack()[0][3] + '()')
    dforig = pd.read_csv(featpath, sep='|')
    lasth = int(dforig.columns[-1][1:])
    hcols = ['h{:03d}'.format(x) for x in range(lasth+1)]

    maxs = np.max(dforig[hcols], axis=1)
    normcols = ['outliersdist', 'avgheight'] + hcols
    dforig[normcols] = dforig[normcols] / maxs[:, None]
    dforig['clsizeavg'] = (dforig.clsize1 + dforig.clsize2) / 2

    ids = np.where(dforig.clsize2 == 0)[0] # compute clsizeavg
    dforig.loc[ids, 'clsizeavg'] = dforig.clsize1

    degr = 3 # poly 3rd degree
    coeffcols = ['coeff{}'.format(i) for i in range(degr, -1, -1)]
    coeffs = np.ndarray((dforig.shape[0], 4), dtype=float)
    residuals = np.ndarray((dforig.shape[0], 4), dtype=float)
    xx = np.array(range(len(hcols)))
    for idx, row in dforig.iterrows():
        yy = row[hcols].values.astype(float)
        coeff = np.polyfit(xx, yy, degr)
        r, _, _, _, _ = np.polyfit(xx, yy, degr, full=True)
        coeffs[idx, :] = coeff
        residuals[idx, :] = r

    for j in range(coeffs.shape[1]):
        dforig[coeffcols[j]] = coeffs[:, j]
        dforig['res'+coeffcols[j]] = residuals[:, j]
        
    nrows = 5; ncols = 2; figscale = 5
    fig, ax = plt.subplots(nrows, ncols,
            figsize=(ncols*figscale, nrows*(.8*figscale)))

    cols = 'outliersdist,avgheight,noutliers,clsizeavg'.split(',')
    for l in ['single']:
        df = dforig[dforig.linkagemeth == l]

        attribs = 'outliersdist,avgheight,noutliers,clsizeavg'.split(',')
        for axidx, attrib in enumerate(attribs):
            aux = axidx
            i = int(aux / 2)
            j = int(aux % 2)
            plot_attrib(df, attrib, palettehex, ax[i, j])
        
        plot_link_heights(df, hcols, palettehex, ax[2, 0])
        plot_fitted_heights(df, len(hcols), coeffcols, palettehex, ax[2, 1])

        t1 = plot_pca(df, cols, [], palettehex, 'Clfeatures ', ax[3, 0])
        t2 = plot_pca(df, hcols, hcols, palettehex, 'HeightsRaw', ax[3, 1])
        t3 = plot_pca(df, cols + hcols, hcols, palettehex, 'ClFeatures+HeightsRaw',
                ax[4, 0])
        t4 = plot_pca(df, cols + coeffcols, [], palettehex,
                'ClFeatures+HeightsFit', ax[4, 1])
        outpath = pjoin(outdir, 'feat_{}_{}.pdf'.format(label, l))
        plt.tight_layout(pad=4.0)
        plt.savefig(outpath)
        labels = attribs + ['heights', 'heightsfit_', 'pca_clfeatures', 'pca_heightsraw',
                'pca_clfeatures_heightsraw', 'pca_clfeatures_heightsfit']
        labels = [label+'_' + l for l in labels]
        utils.export_individual_axis(ax, fig, labels, outdir, [.8, .5, .3, .3],
                'feat_', 'pdf')
        plt.close()

##########################################################
def analyze_features_all(pardir, palettehex, outdir):
    info(inspect.stack()[0][3] + '()')
    featpath = pjoin(pardir, 'features.csv')
    if os.path.exists(featpath):
        print(0)
        analyze_features(featpath, '', palettehex, outdir)
    files = sorted(os.listdir(pardir))
    for f in files:
        dirpath = pjoin(pardir, f)
        if not os.path.isdir(dirpath): continue
        if not os.path.exists(pjoin(dirpath, 'features.csv')): continue
        analyze_features(pjoin(dirpath, 'features.csv'), f, palettehex, outdir)

##########################################################
def analyze_single_precision(pardir, outdir):
    info(inspect.stack()[0][3] + '()')
    featpath = pjoin(pardir, '02d', 'features.csv')
    featdf = pd.read_csv(featpath, sep='|')

    featdf = featdf[featdf.distrib == '1,uniform']
    featdf = featdf[featdf.linkagemeth == 'single']
    n2  = np.count_nonzero(featdf.clsize2)
    n1 = featdf.shape[0] - n2
    return n1, n2

##########################################################
def analyze_ward_precision(pardir, outdir):
    info(inspect.stack()[0][3] + '()')
    files = sorted(os.listdir(pardir))
    n = 0; n2 = 0
    for f in files:
        featpath = pjoin(pardir, f, 'features.csv')
        if not os.path.exists(featpath): continue
        featdf = pd.read_csv(featpath, sep='|')

        unicols = []
        for d in np.unique(featdf.distrib):
            if d.startswith('2,'): unicols.append(d)

        featdf = featdf[featdf.distrib.isin(unicols)]
        featdf = featdf[featdf.linkagemeth == 'ward']
        n2 += np.count_nonzero(featdf.clsize2)
        n += featdf.shape[0]
    n1 = n - n2
    return n1, n2
##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pardir', default='/tmp/',
            help='Path to the folder containing results[all].csv')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.INFO)

    t0 = time.time()
    outdir = pjoin(args.pardir, 'figsresults/')
    iconsdir = pjoin(args.pardir, 'figsarticle/')

    if not os.path.isdir(outdir): os.mkdir(outdir)

    np.random.seed(0)

    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    palettehex2 = palettehex + ['#a66139']

    resdf = concat_results(args.pardir)
    # resdf = filters_by_dim(resdf, [2, 4, 5, 10])

    distribs = np.unique(resdf.distrib)
    linkagemeths = resdf.columns[1:-1]

    plot_parallel_all(resdf, iconsdir, outdir)
    return
    count_method_ranking(resdf, linkagemeths, 'single', outdir)
    for nclusters in ['1', '2']:
        filtered = resdf[resdf['distrib'].str.startswith(nclusters)]
        methscorr = scatter_pairwise(filtered, linkagemeths, palettehex2, outdir)
        plot_meths_heatmap(methscorr, linkagemeths, nclusters, outdir)
        plot_graph(methscorr, linkagemeths, palettehex, nclusters, outdir)

    analyze_features_all(args.pardir, palettehex2, outdir)

    n1, n2 = analyze_single_precision(args.pardir, outdir)
    info('single n1:{} n2:{}'.format(n1, n2))
    n1, n2 = analyze_ward_precision(args.pardir, outdir)
    info('ward n1:{} n2:{}'.format(n1, n2))
    info('Elapsed time:{}'.format(time.time()-t0))
    info('Results are in {}'.format(outdir))

##########################################################
if __name__ == "__main__":
    main()

