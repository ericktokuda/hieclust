#!/usr/bin/env python3
"""Plot parallel coordinates
"""

import argparse
import logging
import time
import os
from os.path import join as pjoin
from logging import debug, info

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D

import pandas as pd
import plotly
import plotly.graph_objects as go

import imageio

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
        loc=[.6, .6],
    )

    ax.tick_params(bottom="off")
    ax.tick_params(axis='x', length=0)

    # axicon = fig.add_axes([0.4,0.4,0.1,0.1])
    # axicon.imshow(np.random.rand(100,100))
    # axicon.set_xticks([])
    # axicon.set_yticks([])

    trans = blended_transform_factory(fig.transFigure, ax.transAxes) # separator
    line = Line2D([0, .95], [-.03, -.03], color='k', transform=trans)
    fig.lines.append(line)

##########################################################
def include_icons(iconpaths, fig):
    for i, iconpath in enumerate(iconpaths):
        im = imageio.imread(iconpath)
        print('im.shape:{}'.format(im.shape))
        newax = fig.add_axes([0.22+i*.074, 0.79, 0.05, 0.2], anchor='NE', zorder=-1)
        newax.imshow(im, aspect='equal')
        newax.axis('off')

##########################################################
def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', help='Results in csv format')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    colours = cm.get_cmap('tab10')(np.linspace(0, 1, 6))
    df = pd.read_csv(args.results, sep='|')
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
    plt.tight_layout(rect=(0.1, 0, 1, 1), h_pad=-3)

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

    plt.savefig(pjoin(args.outdir, 'out.pdf'))
    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

