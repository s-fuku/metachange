import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats
from scipy.special import gammaln

from itertools import permutations

import os

import time

import sys

from volatility_detector import VolatilityDetector

import tqdm

import scaw2 as SCAW2

from smdl import SMDL
from model import Norm1D

from adwin import AdWin

#outdir = '../output/experiment2_final'
#outdir = '../output/experiment2_final_gradual'
#outdir = '../output/experiment2_final_abrupt'
#outdir = '../output/experiment2_final_abrupt_20191001'
outdir = '../output/experiment2_final_abrupt_20191004_v2'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def generate_data_experiment2(length=100, n_repeat=10, ymax=1.0, sigma=0.1, seed=123):
    np.random.seed(seed)
    
    X_list = []
    for i in range(n_repeat):
        X1 = sigma * np.random.randn(length)
        X_list.append(X1)
        
        X2 = ymax + sigma * np.random.randn(length)
        X_list.append(X2)
    
    #X3 = ymax - ymax * np.arange(1, length + 1) / length + sigma * np.random.randn(length)
    X3 = ymax * np.arange(1, length + 1) / length + sigma * np.random.randn(length)
    X_list.append(X3)
    
    X4 = sigma * np.random.randn(3*length)
    X_list.append(X4)
    
    #X5 = np.arange(1, length + 1) / length + sigma * np.random.randn(length)
    #X_list.append(X5)
    
    #X6 = ymax + sigma * np.random.randn(length)
    #X_list.append(X6)
    
    X = np.hstack(X_list)
    X = X.reshape(-1, 1)
    return X

def generate_data_experiment2_gradual(length=500, n_repeat=50, 
                              y_max=1.0, sigma=0.1, seed=123):
    np.random.seed(seed)
    
    X_list = []
    
    # SEGMENT1
    for i in range(n_repeat):
        X1 = sigma * np.random.randn(length)
        X_list.append(X1)
        
        X2 = y_max * np.arange(1, length + 1)/ length + sigma * np.random.randn(length)
        X_list.append(X2)
        
        X3 = y_max + sigma * np.random.randn(length)
        X_list.append(X3)
        
        X4 = y_max - y_max * np.arange(1, length + 1)/length + sigma * np.random.randn(length)
        X_list.append(X4)
        
    # SEGMENT2
    X5 = sigma * np.random.randn(length)
    X_list.append(X5)
    X6 = y_max + sigma * np.random.randn(length)
    X_list.append(X6)
    X7 = sigma * np.random.randn(length)
    X_list.append(X7)
    X8 = y_max + sigma * np.random.randn(length)
    X_list.append(X8)

    X = np.hstack(X_list)
    X = X.reshape(-1, 1)
    return X

def detect_change_points_maximum(scores):
    scores = np.array(scores)
    idxes_stats_positive = np.where(scores > 0)[0]

    idxes_end = np.where(
        np.diff(idxes_stats_positive) > 1
    )[0]
    end = idxes_stats_positive[idxes_end]

    idxes_start = idxes_end + 1
    if 0 not in idxes_start:
        idxes_start = np.hstack((0, idxes_start))

    start = idxes_stats_positive[idxes_start]

    if idxes_stats_positive[idxes_start[1] - 1] not in end:
        end = np.hstack((idxes_stats_positive[idxes_start[1] - 1], end))
    if idxes_stats_positive[-1] not in end:
        end = np.hstack((end, idxes_stats_positive[-1]))

    change_points = []
    for s, e in zip(start, end):
        cp = s + np.argmax(scores[s:e + 1])
        change_points.append(cp)

    change_points = np.array(change_points)
    return change_points


def calc_auc(fars, benefits):
    # sort by ascending order
    #idx_ordered = np.argsort(fars)
    idx_ordered = np.lexsort((fars, benefits))
    fars_ordered = fars[idx_ordered]
    benefits_ordered = benefits[idx_ordered]
    #if abs(fars_ordered[0]) > 1e-6:
    #if fars_ordered[0] != 0.0:
    if np.abs(fars_ordered[0]) > 1e-6:
        #fars_ordered = np.hstack((0, 0, fars_ordered))
        fars_ordered = np.hstack((0, 0, fars_ordered))
        #benefits_ordered = np.hstack((0, benefits_ordered[0], benefits_ordered))
        #fars_ordered = np.hstack((0, fars_ordered[0], fars_ordered))
        #benefits_ordered = np.hstack((0, 0, benefits_ordered))
        benefits_ordered = np.hstack((0, benefits_ordered[0], benefits_ordered))
        #fars_ordered = np.hstack((0, fars_ordered))
        #benefits_ordered = np.hstack((0, benefits_ordered))
    elif benefits_ordered[0] != 0.0:
        fars_ordered = np.hstack((0, fars_ordered))
        benefits_ordered = np.hstack((0, benefits_ordered))

    # calculate AUC
    #auc = np.abs(np.sum(np.diff(fars_ordered/np.max(fars_ordered)) *
    #                       np.abs(benefits_ordered[:-1])/np.max(np.abs(benefits_ordered[:-1])))
    # )
    if np.all(benefits_ordered == 0):
        auc = 0.0
    else:
        auc = np.trapz(benefits_ordered,
                   fars_ordered/np.max(fars_ordered))
    return auc


def detect_change_points_start(scores):
    scores = np.array(scores)
    idxes_stats_positive = np.where(scores > 0)[0]

    idxes_end = np.where(
        np.diff(idxes_stats_positive) > 1
    )[0]
    end = idxes_stats_positive[idxes_end]

    idxes_start = idxes_end + 1
    if 0 not in idxes_start:
        idxes_start = np.hstack((0, idxes_start))

    start = idxes_stats_positive[idxes_start]

    if idxes_stats_positive[idxes_start[1] - 1] not in end:
        end = np.hstack((idxes_stats_positive[idxes_start[1] - 1], end))
    if idxes_stats_positive[-1] not in end:
        end = np.hstack((end, idxes_stats_positive[-1]))

    change_points = np.array(start)
    return change_points


def calc_metachange_state(X, change_points, h=100, mu_max=2.0, sigma_min=0.005):
    metachange_stats = []

    # for t in range(h, len(X)-h):
    for i, cp in enumerate(change_points):
        mean1 = np.mean(X[(cp - h):cp, :].ravel())
        std1 = np.std(X[(cp - h):cp, :].ravel())
        # mean2 = np.mean(X[cp:(cp+h+1), :].ravel())
        # std2 = np.std(X[cp:(cp+h+1), :].ravel())
        mean2 = np.mean(X[(cp+1):(cp+h+1), :].ravel())
        std2 = np.std(X[(cp+1):(cp+h+1), :].ravel())

        if i == 0:
            mean1_prev, std1_prev = mean1, std1
            mean2_prev, std2_prev = mean2, std2
            continue

        metachange_up = np.mean(
            -scipy.stats.norm(mean1 + (mean2_prev - mean1_prev), std1 + (std2_prev - std1_prev)).logpdf(
                X[(cp+1):(cp+h+1), :].ravel()))
        metachange_down = np.mean(
            -scipy.stats.norm(mean1 - (mean2_prev - mean1_prev), std1 - (std2_prev - std1_prev)).logpdf(
                X[(cp+1):(cp+h+1), :].ravel()))

        metachange = np.nanmin([metachange_up, metachange_down])

        metachange_stats.append(metachange)
        # print(metachange)

        mean1_prev, std1_prev = mean1, std1
        mean2_prev, std2_prev = mean2, std2

    return np.array(metachange_stats)
    # return np.abs(np.diff(metachange_stats))


def calc_metachange_time_md(X, change_points, r=0.05):
    change_points_diff = np.diff(change_points)

    lambdas_hat = np.array(
        [(1 - (1 - r) ** (i + 1)) / \
         (r * np.sum((1 - r) ** np.arange(i, -1, -1) * change_points_diff[:i + 1])) \
         for i in range(len(change_points_diff))]
    )
    codelen = -np.log(lambdas_hat[:-1]) + lambdas_hat[:-1] * change_points_diff[1:]
    change_rate_codelen = np.diff(codelen) / codelen[:-1]

    return codelen


def calc_metachange_time_vd(X, change_points, r=0.05, seed=0):
    change_points_diff = np.diff(change_points)

    vdetector = VolatilityDetector(seed=i)
    relvol = np.array([vdetector.detect(change_points_npa_diff[i]) for i in range(len(change_points_npa_diff))])

    return relvol


def calc_benfit_far_mcat(change_rate_codelen, change_points,
                         length1, length2, n_repeat,
                         limit=5, r=0.2, B=32, R=32):
    ## MCAT
    benefits_md_i, fars_md_i = [], []
    thr_list = np.sort(np.abs(change_rate_codelen[B + R - 3:]))
    thr_list = np.linspace(thr_list[0], thr_list[-1], 100)

    for thr in thr_list:
        # metachange detector
        idxes_over_thr_after_cp_within = np.where(
            np.logical_and(
                np.logical_and(
                    change_points[B + R:] >= 2 * length1 * n_repeat,  # + blockSize,
                    change_points[B + R:] <= 2 * length1 * n_repeat + limit * length2
                ),
                np.abs(change_rate_codelen[B + R - 3:]) > thr
            ))[0]
        if len(idxes_over_thr_after_cp_within) == 0:
            benefit = 0.0
        else:
            benefit = 1 - (change_points[B + R + idxes_over_thr_after_cp_within[0]] - 2 * length1 * n_repeat) / (
            limit * length2)

        # false positive rate
        n_fp = np.sum(
            np.logical_and(
                np.logical_or(
                    change_points[B + R:] >= 2 * length1 * n_repeat + limit * length2,
                    change_points[B + R:] < 2 * length1 * n_repeat
                ),
                np.logical_and(
                    np.abs(change_rate_codelen[B + R - 3:]) > thr,
                    ~np.isnan(change_rate_codelen[B + R - 3:])
                )
            )
        )

        benefits_md_i.append(benefit)
        fars_md_i.append(n_fp)

    return benefits_md_i, fars_md_i


def calc_benfit_far_vd(change_points, change_points_diff,
                       length1, length2, n_repeat, limit=5,
                       blockSize=32, seed=0):
    cps = change_points[1:]
    ## volatility detector
    vdetector = VolatilityDetector(seed=seed)
    relvol = np.array([vdetector.detect(change_points_diff[j]) for j in range(len(change_points_diff))])

    benefits_vd_i, fars_vd_i = [], []
    bnd_list = np.sort(np.abs(relvol[~np.isnan(relvol)] - 1.0))
    bnd_list = np.linspace(bnd_list[0], bnd_list[-1], 100)
    for bnd in bnd_list:
        is_metachange = np.logical_or(relvol > 1.0 + bnd, relvol < 1.0 - bnd)

        idxes_over_thr = np.where(
            np.logical_and(
                np.logical_or(relvol > 1.0 + bnd, relvol < 1.0 - bnd),
                ~np.isnan(relvol)
            )
        )[0]
        cps_over_thr = cps[idxes_over_thr]
        within_tol_interval = np.logical_and(
            cps_over_thr - 2 * n_repeat * length1 >= blockSize,
            cps_over_thr - 2 * n_repeat * length1 <= limit * length2)

        # benefit
        benefit = 0.0
        if np.any(within_tol_interval):
            dist_from_cp = np.abs(cps_over_thr[within_tol_interval] - 2 * n_repeat * length1)
            benefit = 1 - (dist_from_cp[0] / (limit * length2))

        benefits_vd_i.append(benefit)

        # false alarm
        n_fp = np.sum(
            np.logical_and(
                np.logical_and(
                    np.logical_or(relvol > 1.0 + bnd, relvol < 1.0 - bnd),
                    ~np.isnan(relvol)
                ),
                np.logical_or(
                    cps <= 2 * n_repeat * length1,
                    cps >= 2 * n_repeat * length1 + limit * length2
                )
            )
        )
        fars_vd_i.append(n_fp)

    return benefits_vd_i, fars_vd_i



def calc_metachange_stats_v2(X, change_points, h=100, mu_max=2.0, sigma_min=0.005):
    metachange_stats = []

    # for t in range(h, len(X)-h):
    for i, cp in enumerate(change_points):
        mean1 = np.mean(X[(cp - h):cp, :].ravel())
        std1 = np.std(X[(cp - h):cp, :].ravel())
        # mean2 = np.mean(X[cp:(cp+h+1), :].ravel())
        # std2 = np.std(X[cp:(cp+h+1), :].ravel())
        mean2 = np.mean(X[(cp + 1):(cp + h + 1), :].ravel())
        std2 = np.std(X[(cp + 1):(cp + h + 1), :].ravel())

        if i == 0:
            mean1_prev, std1_prev = mean1, std1
            mean2_prev, std2_prev = mean2, std2
            continue

        metachange_up = np.mean(
            -scipy.stats.norm(mean1 + (mean2_prev - mean1_prev), std1 + (std2_prev - std1_prev)).logpdf(
                X[(cp+1):(cp+h+1), :].ravel()))
        metachange_down = np.mean(
            -scipy.stats.norm(mean1 - (mean2_prev - mean1_prev), std1 - (std2_prev - std1_prev)).logpdf(
                X[(cp+1):(cp+h+1), :].ravel()))

        metachange = np.nanmin([metachange_up, metachange_down]) - 0.5 * np.log((16 * np.abs(mu_max))/ (np.pi * sigma_min**2)) \
                        +np.log(2.0) + 1.0 - gammaln(0.5)

        metachange_stats.append(metachange)
        # print(metachange)

        mean1_prev, std1_prev = mean1, std1
        mean2_prev, std2_prev = mean2, std2

    return np.array(metachange_stats)
    # return np.abs(np.diff(metachange_stats))


def calc_benefit_far_state(change_points, stats, metapoint, length, n_repeat, limit=5):
    benefits = []
    fars = []

    thresholds = np.sort(stats[~np.isnan(stats)]) - 1e-3
    thresholds = np.linspace(thresholds[0], thresholds[-1], 100)

    for thr in thresholds:
        idxes_over_thr = np.where(stats >= thr)[0]
        within_tol_interval = np.logical_and(0 <= (change_points[idxes_over_thr] - metapoint),
                                             (change_points[idxes_over_thr] - metapoint) <= limit * length)
        # benefit
        benefit = 0.0
        if np.any(within_tol_interval):
            # dist_from_cp = np.abs(change_points[idxes_over_thr][within_tol_interval] - metapoint)
            # benefit = 1 - dist_from_cp[0]/(limit*length)
            
            #print(change_points[idxes_over_thr][within_tol_interval][0])
            benefit = 1 - (change_points[idxes_over_thr][within_tol_interval][0] - metapoint) / (limit * length)

        # false positive rate
        n_fp = np.sum(
            np.logical_and(
                np.logical_or(
                    change_points <= metapoint,
                    change_points >= metapoint + limit * length
                ),
                np.logical_and(
                    stats >= thr,
                    ~np.isnan(stats)
                )
            )
        )

        benefits.append(benefit)
        fars.append(n_fp)

    fars = np.array(fars)
    benefits = np.array(benefits)

    #print(benefits)
    #print(fars)

    return benefits, fars

def gridsearch_ad(
        X, 
        length,
        real_changepoints,
        delta,
        n_repeat=50,
        ymax=1.0,
        delay=100,
        delay_cp=30, 
        n_trial=30
):
    aucs = []
    T = len(X)

    scores = []
    # Adwin
    ad = AdWin(delta=delta)

    change_points = []
    scores = []
    for t, x in enumerate(X.ravel()):
        prev_lenwin = ad.window_len
        if ad.set_input(x):
            change_points.append(t)
            post_lenwin = ad.window_len
            score = prev_lenwin - post_lenwin + 1
            scores.append(score)
        else:
            scores.append(0.0)

    change_points = np.array(change_points)
    scores = np.array(scores)

    thresholds = np.sort(scores[~np.isnan(scores)])
    thresholds = np.linspace(thresholds[0], thresholds[-1], 100)

    fars, benefits = [], []
    for thr in thresholds:
        idxes_over_thr = np.where(scores >= thr)[0]
        n_fp = len(idxes_over_thr)

        benefit = 0.0
        count = 0
        for r_cp in real_changepoints:
            ok = np.logical_and(idxes_over_thr - r_cp >= 0,
                                idxes_over_thr - r_cp <= delay)
            if any(ok):
                benefit += 1 - (idxes_over_thr[ok][0] - r_cp) / delay
                count += 1
            #if count >= 1:
            #    benefit /= count

        fars.append(n_fp)
        benefits.append(benefit)

    #print(fars)
    #print(benefits)
    
    fars = np.array(fars)
    benefits = np.array(benefits)

    auc = calc_auc(fars, benefits/np.max(benefits))
    aucs.append(auc)
    
    tp = 0
    for cp in real_changepoints:
        ok = np.abs(change_points - cp) <= delay_cp
        if np.any(ok):
            tp += 1
            
    m = len(change_points)
    l = len(real_changepoints)
    fp = m - tp
    fn = l - tp
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    
    diff = change_points - 2*n_repeat*length
    idxes = np.where(diff >= 0)[0]
    d = diff[idxes[0]]
    
    return aucs, change_points, precision, recall, d, tp, fp, fn

def gridsearch_mcd(
         X, 
         length,
         change_points, 
         metapoint, 
         h=50,
         epsilon=0.1,
         n_repeat=10,
         ymax=1.0,  
         limit=1, 
         n_trial=20
): 
    # AUC for metachange
    mcas = calc_metachange_stats_v2(X, change_points, h=h)
    #metapoint = 4*n_repeat*length + length
    benefits_mcas_i, fars_mcas_i = calc_benefit_far_state(
                                       change_points[1:], mcas, 
                                       metapoint, length, 
                                       n_repeat, limit=limit)
    auc_mcas = calc_auc(np.array(fars_mcas_i), 
                        np.array(benefits_mcas_i))

    print('## SMDL-MC')
    T = len(X)
    #smdl = SMDL(h, T, Norm1D, 0.05)
    smdl = SMDL(Norm1D, 0.05)
    #mdl_stats = [np.nan] * w + \
    #            [smdl.calc_change_score(X[(i-h):(i+h), :].ravel(), 
    #            mu_max=2.0, sigma_min=0.005) \
    #            for i in range(h, T-h)] + \
    #            [np.nan] * h
    mdl_stats = [smdl.calc_change_score(X[(i-h):(i+h+1), :].ravel(), h, 
                 mu_max=2.0, sigma_min=0.005) \
                 for i in change_points]
    mdl_stats = np.array(mdl_stats)

    #mdl_stats_rate = np.abs(np.diff(mdl_stats[change_points])/
    #                                mdl_stats[change_points][:-1])
    mdl_stats_rate = np.abs(np.diff(mdl_stats)/ mdl_stats[:-1])
    benefits_smdlmc, fars_smdlmc = calc_benefit_far_state(
                                           change_points[1:], mdl_stats_rate,
                                           metapoint, length,
                                           n_repeat, limit=limit)

    auc_smdlmc = calc_auc(np.array(fars_smdlmc), 
                          np.array(benefits_smdlmc))

    return auc_mcas, auc_smdlmc

results = []

n_repeat = 10
n_trial = 50
for length in [500, 1000, 2000]:
    real_changepoints = np.arange(length, 2*n_repeat*length + 2*length, length)
    #real_changepoints = np.linspace(length, 4*n_repeat*length + 4*length, length)
    for i in tqdm.tqdm(range(n_trial)):
        X = generate_data_experiment2(length, n_repeat=n_repeat, ymax=0.5, seed=i)
        #X = generate_data_experiment2_gradual(length, n_repeat=n_repeat, seed=i)
        metapoint = 2*n_repeat*length
        #metapoint = 4*n_repeat*length + length
        #for w in [int(0.1*length), int(0.2*length), int(0.3*length)]:
        for delta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            #try:
            auc_cp, change_points, precision, recall, delay, tp, fp, fn = gridsearch_ad(
                     X, 
                     length,
                     real_changepoints,
                     delta=delta,
                     n_repeat=n_repeat)
            #except:
            #    continue
                
            #for w in [50, 100, 150]:
            for w in [int(0.1*length), int(0.2*length)]:
                try:
                    auc_mcas, auc_smdlmc = gridsearch_mcd(
                        X, 
                        length,
                        change_points, 
                        metapoint, 
                        h=w,
                        n_repeat=n_repeat
                    ) 

                    print('length:', length,
                          'w:', w,
                          'delta:', delta,
                          'AUC_CP:', auc_cp, 
                          'PRECISION:', precision,
                          'RECALL:', recall,
                          'DELAY:', delay, 
                          'TP:', tp,
                          'FP:', fp,
                          'FN:', fn,
                          'AUC_MCAS:', auc_mcas,
                          'AUC_SMDLMC:', auc_smdlmc
                    )
                    result = pd.DataFrame({
                        'length': length,
                        'w': w,
                        'delta': delta,
                        'AUC_CP': auc_cp,
                        'PRECISION_CP': precision,
                        'RECALL_CP': recall, 
                        'DELAY': delay,
                        'TP': tp,
                        'FP': fp,
                        'FN': fn, 
                        'AUC_MCAS': auc_mcas, 
                        'AUC_SMDLMC': auc_smdlmc
                    }, columns=['length', 'w', 'delta', 
                                'AUC_CP', 'PRECISION_CP', 'RECALL_CP', 'DELAY',  
                                'TP', 'FP', 'FN', 
                                'AUC_MCAS', 'AUC_SMDLMC'])
                    results.append(result)
                except:
                    continue

                results_df = pd.concat(results)
                #results_df.to_csv(os.path.join(outdir, 'adwin_auc.csv'), index=None)
                results_df.to_csv(os.path.join(outdir, 'adwin_auc_mixed.csv'), index=None)

#results_df = pd.concat(results)
#results_df.to_csv(os.path.join(outdir, 'adwin_auc.csv'), index=None)
#results_df.to_csv(os.path.join(outdir, 'adwin_auc.csv'), index=None)
