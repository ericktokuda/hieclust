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
MAXK = 5

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
def define_pred_vectors(rels, nclu, distribs, maxnclu):
    sz1, sz2, sz3 = rels.shape
    vpred = np.zeros((sz1, sz2, maxnclu))
    for i in range(sz1):
        for j in range(sz2):
            for k in range(maxnclu):
                inds = np.where(nclu[i, j, :] == k + 1)
                r = rels[i, j, :][inds]
                if len(r) == 0: continue
                vpred[i, j, k] = np.sum(r)
    return vpred

##########################################################
def validate_vpred(vpred, gtruths):
    sz1, sz2, sz3 = vpred.shape
    diffnorms = np.zeros((sz1, sz2), dtype=float) # (distribs, linkagemeths)
    for i in range(sz1):
        for j in range(sz2):
            diffnorms[i, j] = np.linalg.norm(vpred[i, j, :] - gtruths[i, :])
    return diffnorms

##########################################################
def export_results(diffnorms, vpred, rels, k, distribs, datadim, linkagemeths,
        clrelsize, prunparam, outdir):
    datapred = []
    datadiff = []
    for i, d in enumerate(distribs):
        m, dd, p = d.split(',')
        for j, l in enumerate(linkagemeths):
            row = [m, dd, p, datadim, l, clrelsize, prunparam]
            row += vpred[i, j, :].tolist()
            datapred.append(row)
            datadiff.append([m, dd, p, datadim, l, clrelsize,
                prunparam, diffnorms[i, j]])

    cols = ['nmodes', 'distrib', 'distribparam', 'datadim', 'linkagemeth',
            'clrelsize', 'prunparam', 'diffnorm']
    fpath = pjoin(outdir, 'diffnorms.csv')
    df = pd.DataFrame(datadiff, columns=cols)
    df.to_csv(fpath, index=False, float_format='%.3f')

    fpath = pjoin(outdir, 'vpred.csv')
    cols = ['nmodes', 'distrib', 'distribparam', 'datadim', 'linkagemeth',
            'clrelsize', 'prunparam']
    for i in range(k): cols.append('clu{}'.format(i+1))
    
    df = pd.DataFrame(datapred, columns=cols)
    df.to_csv(fpath, index=False, float_format='%.3f')

##########################################################
def export_features(diffnorms, rels, distribs, linkagemeths, outdir):
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
def dummy_data(distribs, samplesz, datadim):
    data = {}
    partsz = {}
    # d = [21, 22, 23, 24, 25, 40, 42, 44, 46, 48]
    # data['2,gaussian,4'] = [[dd] for dd in d]
    # partsz['2,gaussian,4'] = [5, 5]

    d = [21, 22, 23, 42, 44, 46, 61, 62, 63, 64]
    data['3,gaussian,4'] = [[dd] for dd in d]
    partsz['3,gaussian,4'] = [3, 3, 4]
    return data, partsz
##########################################################
def run_all_experiments(linkagemeths, datadim, samplesz, distribs, k, clrelsize,
                        c, precthresh, metric, nrealizations, outdir):
    info(inspect.stack()[0][3] + '()')

    errdir = pjoin(outdir, 'errors')
    os.makedirs(errdir, exist_ok=True)

    clsize = int(clrelsize * samplesz)
    nfeats = 6 + (samplesz - 1)
    errors = []

    distribs = sorted(distribs); linkagemeths = sorted(linkagemeths)
    ndistribs, nlinkagemeths = len(distribs), len(linkagemeths)

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
        # data, partsz = dummy_data(distribs, samplesz, datadim)
        # ut.plot_data(data, partsz, outdir); return

        for j, linkagemeth in enumerate(linkagemeths): # Loop-method
            for i, distrib in enumerate(data): # Loop-distrib
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

                if len(clids) == 0: continue #TODO: DEFINE WHAT TO when errors

                nclu[i][j][r] = len(clids)
                rels[i][j][r] = ut.calculate_relevance(z, clids)

                # prec = ut.compute_max_precision(clids, partsz[distrib], z)
                f = extract_features(c, rels[i][j][r], len(outliers), clids, z)
                feats[distrib][linkagemeth][r] = f
                # TODO: What to do do when precision below precthresh???
                # nimprec[distrib][linkagemeth] += nimprec

    gtruths = ut.compute_gtruth_vectors(k, distribs, nrealizations)
    vpred = define_pred_vectors(rels, nclu, distribs, k)

    diffnorms = validate_vpred(vpred, gtruths)
    export_results(diffnorms, vpred, rels, k, distribs, datadim, linkagemeths,
            clrelsize, c, outdir)
    # export_features(diffnorms, rels, distribs, linkagemeths, outdir)

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
    shutil.copy(args.config, pjoin(c['outdir'], 'config.json'))
    np.random.seed(c['seed']); random.seed(c['seed'])

    run_all_experiments(c['linkagemeths'].split('|'), c['datadim'], c['samplesz'],
         c['distribs'].split('|'), c['maxnclu'], c['clrelsize'], c['pruningparam'],
         c['precthresh'], c['metric'], c['nrealizations'],
         c['outdir'])

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(c['outdir']))
