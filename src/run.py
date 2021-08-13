#!/usr/bin/env python3
"""Experimental study of traditional hierarchical clustering methods."""

from os.path import join as pjoin
import os, sys, time, argparse, inspect
import datetime, json, shutil
import numpy as np

import scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
import pandas as pd
import utils
from myutils import info, create_readme

import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt; plt.style.use('ggplot')

##########################################################
def extract_features(ouliersdist, avgheight, noutliers, clustids, z):
    maxdist = z[-1, 2]
    clsizes = [0] * 2
    for i in [0, 1]:
        if len(clustids) > i:
            clsizes[i] = len(utils.get_leaves(z, clustids[i]))

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
                        outliersratio, precthresh, metric, nrealizations,
                        seed, outdir):
    info(inspect.stack()[0][3] + '()')

    clsize = int(clrelsize * samplesz)
    nfeats = 6 + (samplesz - 1)

    # Initialize variables
    rels = {}; methprec = {}; nimprec = {}; features = {}
    for distrib in distribs:
        rels[distrib] = {}; methprec[distrib] = {};
        nimprec[distrib] = {}; features[distrib] = {}
        for l in linkagemeths:
            rels[distrib][l] = [[], []] # Relevances
            methprec[distrib][l] = [] # Precision
            nimprec[distrib][l] = 0 # Number of cases below precthresh
            features[distrib][l] = np.zeros((nrealizations, nfeats))

    for r in range(nrealizations): # Loop-realization
        info('Realization {:02d}'.format(r))
        data, partsz = utils.generate_data(distribs, samplesz, datadim)
        # utils.plot_data(data, partsz, outdir); return

        for j, linkagemeth in enumerate(linkagemeths): # Loop-method
            for i, distrib in enumerate(data): # Loop-distrib
                npred, rel, feats, prec = \
                    utils.find_clusters(distrib, z, clsize, k, outliersratio)
                # try:
                    # z = linkage(data[distrib], linkagemeth, metric)
                # except exception as e:
                    # filename = 'error_{}_{}.npy'.format(distrib, linkagemeth)
                    # np.save(pjoin(outdir, filename), data[distrib])
                    # raise(e)

                # k = int(distrib.split(',')[0])
                # clustids, rel, ouliersdist, outliers = \
                    # utils.find_clusters(z, clsize, k, outliersratio)
                # features[distrib][linkagemeth][r] = \
                    # extract_features(ouliersdist, rel,
                                     # len(outliers), clustids, z)

                # prec = utils.compute_max_precision(clustids, partsz[distrib], z)
                # ngtruth = int(distrib.split(',')[0])
                # npred = len(clustids)

                # if ngtruth == npred and prec < precthresh: # prec limiarization
                    # npred = (npred % 2) + 1
                    # nimprec[distrib][linkagemeth] += 1

                rels[distrib][linkagemeth][npred-1].append(rel)
                features[distrib][linkagemeth][r] = feats
                # TODO: What to do do when precision below precthresh???
                # nimprec[distrib][linkagemeth] += nimprec

    avgrel = utils.average_relevances(rels, distribs, linkagemeths)
    filename = pjoin(outdir, 'nimprec.csv')
    pd.DataFrame(nimprec).to_csv(filename, sep='|', index_label='linkagemeth')

    gtruths = utils.compute_gtruth_vectors(distribs, nrealizations)
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

    run_all_experiments(c['linkagemeths'].split('|'), c['datadim'], c['samplesz'],
         c['distribs'].split('|'), c['clrelsize'], c['pruningparam'],
         c['precthresh'], c['metric'], c['nrealizations'], c['seed'],
         c['outdir'])

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(c['outdir']))
