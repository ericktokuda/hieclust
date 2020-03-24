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
import pandas as pd
import plotly
import plotly.graph_objects as go


##########################################################
def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('results', help='results file in csv format')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    for d in sorted(os.listdir('./')):
        if not '20200318-hieclust_' in d: continue
        dim = d.replace('20200318-hieclust_', '')
        plot_parallel(pjoin(d, 'results.csv'), pjoin(args.outdir, 'error_' + dim + '.pdf'))
    info('Elapsed time:{}'.format(time.time()-t0))

def plot_parallel(resultscsv, outfile):
    colours = cm.get_cmap('tab10')(np.linspace(0, 1, 6))
    df = pd.read_csv(resultscsv)
    fig, ax = plt.subplots(figsize=(7,5))

    ax = pd.plotting.parallel_coordinates(
        df, 'linkagemeth',
        axvlines_kwds={'visible':True, 'color':np.ones(3)*.6,
                       'linewidth':4, 'alpha': 0.9, },
        ax=ax, linewidth=4, alpha=0.9,
        color = colours,
    )
    ax.yaxis.grid(False)
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_xticklabels(list('ABCDEFGHIJK'))
    ax.set_xlim(-.5, 10)
    ax.set_ylabel('Accumulated error')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc=[.6, .6])

    ax.tick_params(bottom="off")
    plt.tight_layout()
    plt.savefig(outfile)

if __name__ == "__main__":
    main()

