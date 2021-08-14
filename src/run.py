#!/usr/bin/env python3
"""Experimental study of traditional hierarchical clustering methods."""

from os.path import join as pjoin
import os, sys, time, random, argparse, inspect
import datetime, json, shutil
import numpy as np

import scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
import pandas as pd
import utils as ut
from myutils import info, create_readme

import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt; plt.style.use('ggplot')

##########################################################
def extract_features(ouliersdist, avgheight, noutliers, clustids, z):
    maxdist = z[-1, 2]
    clsizes = [0] * 2
    for i in [0, 1]:
        if len(clustids) > i:
            clsizes[i] = len(ut.get_leaves(z, clustids[i]))

    features = np.array([maxdist, ouliersdist, avgheight, noutliers] + clsizes)
    features = np.concatenate((features, z[:, 2]))
    return features

##########################################################
def compute_rel_to_gtruth_difference(accrel, gtruths, distribs, linkagemeths,
        nrealizations):
    diff = {} # difference to the ground-truth
    diffnorms = {}
    for k in distribs:
        diff[k] = dict((el, np.zeros(2)) for el in linkagemeths)
        diffnorms[k] = {}

    for i, distrib in enumerate(distribs):
        for j, linkagemeth in enumerate(linkagemeths):
            diff[distrib][linkagemeth] = gtruths[distrib] - accrel[distrib][linkagemeth]
            diffnorms[distrib][linkagemeth] = np.linalg.norm(diff[distrib][linkagemeth])

    winners = {}
    for d in distribs:
        minvalue = 1000
        for l in linkagemeths:
            if diffnorms[d][l] < minvalue:
                winners[d] = l
                minvalue = diffnorms[d][l]
    return diffnorms, winners

##########################################################
def export_results(diffnorms, rels, features, distribs, linkagemeths, ndims, outdir):
    df = pd.DataFrame.from_dict(diffnorms, orient='index')
    df['dim'] = pd.Series([ndims for x in range(len(df.index))], index=df.index)
    df.to_csv(pjoin(outdir, 'results.csv'), sep='|', index_label='distrib')

    nrealizations, nfeats = features[distribs[0]][linkagemeths[0]].shape
    df = pd.DataFrame.from_dict(features, orient='index')

    fh = open(pjoin(outdir, 'features.csv'), 'w')
    header = 'distrib|linkagemeth|realiz|maxdist|outliersdist|avgheight|noutliers|clsize1|clsize2|'
    header += '|'.join(['h{:03d}'.format(x) for x in range(nfeats - 6)])
    print(header, file=fh)
    for l in linkagemeths:
        for d in distribs:
            for r in range(nrealizations):
                s = '{}|{}|{}|'.format(d, l, r)
                s += ('|'.join([str(x) for x in df[l][d][r]]))
                print(s, file=fh)
    fh.close()

##########################################################
def run_all_experiments(linkagemeths, datadim, samplesz, distribs, clrelsize,
                        c, precthresh, metric, nrealizations, outdir):
    info(inspect.stack()[0][3] + '()')

    errdir = pjoin(outdir, 'errors')
    os.makedirs(errdir, exist_ok=True)

    clsize = int(clrelsize * samplesz)
    nfeats = 6 + (samplesz - 1)
    errors = []

    distribs = sorted(distribs); linkagemeths = sorted(linkagemeths)
    ndistribs, nlinkagemeths = len(distribs), len(linkagemeths)

    # Initialize variables
    # dinds = {k: v for v, k in enumerate(distribs)}
    # dkeys = dict(enumerate(distribs))
    # linds = {k: v for v, k in enumerate(linkagemeths)}
    # lkeys = dict(enumerate(linkagemeths))

    nclu = np.zeros((ndistribs, nlinkagemeths, nrealizations), dtype=int)
    rels = np.zeros((ndistribs, nlinkagemeths, nrealizations), dtype=float)

    feats = {}
    for distrib in distribs:
        feats[distrib] = {}
        for l in linkagemeths:
            feats[distrib][l] = np.zeros((nrealizations, nfeats))

    for r in range(nrealizations): # Loop-realization
        info('Realization {:02d}'.format(r))
        data, partsz = ut.generate_data(distribs, samplesz, datadim)
        # ut.plot_data(data, partsz, outdir); return

        for j, linkagemeth in enumerate(linkagemeths): # Loop-method
            for i, distrib in enumerate(data): # Loop-distrib
                k = int(distrib.split(',')[0]) # Queried number of clusters
                d = data[distrib]

                # try:
                    # z, clustids, outliers = ut.find_clusters(d, k, clsize, c)
                # except Exception as e:
                    # suff = '{};{};{}'.format(distrib, linkagemeth, r)
                    # np.save(pjoin(errdir, suff + '.npy'), d)
                    # errors.append(suff)
                    # continue

                z, clids, outliers = ut.find_clusters(d, k, linkagemeth,
                                                         metric, clsize, c)
                m = len(clids)
                if m == 0: continue #TODO: DEFINE WHAT TO when errors

                nclu[i][j][r] = m
                rels[i][j][r] = ut.calculate_relevance(z, clids)

                # prec = ut.compute_max_precision(clids, partsz[distrib], z)
                features = extract_features(c, rels[i][j][r], len(outliers), clids, z)

                feats[distrib][linkagemeth][r] = features
                # TODO: What to do do when precision below precthresh???
                # nimprec[distrib][linkagemeth] += nimprec

    breakpoint()
    avgrel = ut.average_relevances(rels, distribs, linkagemeths)
    filename = pjoin(outdir, 'nimprec.csv')
    pd.DataFrame(nimprec).to_csv(filename, sep='|', index_label='linkagemeth')

    gtruths = ut.compute_gtruth_vectors(distribs, nrealizations)
    diffnorms, winners = compute_rel_to_gtruth_difference(
            avgrel, gtruths, distribs, linkagemeths, nrealizations)

    export_results(diffnorms, rels, features, distribs, linkagemeths,
                   datadim, outdir)

##########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Config file (.json)')
    args = parser.parse_args()

    info(datetime.date.today())
    t0 = time.time()

    if not os.path.exists(args.config): raise Exception('Could not read config file')
    c = json.load(open(args.config))

    os.makedirs(c['outdir'], exist_ok=True)
    readmepath = create_readme(sys.argv, c['outdir'])
    shutil.copy(args.config, c['outdir'])
    np.random.seed(c['seed']); random.seed(c['seed'])

    run_all_experiments(c['linkagemeths'].split('|'), c['datadim'], c['samplesz'],
         c['distribs'].split('|'), c['clrelsize'], c['pruningparam'],
         c['precthresh'], c['metric'], c['nrealizations'],
         c['outdir'])

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(c['outdir']))