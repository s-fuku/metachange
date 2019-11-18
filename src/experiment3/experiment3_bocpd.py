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

import tqdm

from smdl import SMDL
from model import Norm1D

from mybocpd import BOCD, constant_hazard, StudentT

from functools import partial

from volatility_detector import VolatilityDetector

#outdir = '../output/experiment3_final_fixed'
#outdir = '../output/experiment3_final_fixed_gradual'
#outdir = '../output/experiment3_final_fixed_abrupt'
#outdir = '../output/experiment3_final_abrupt'
#outdir = '../output/experiment3_final_abrupt_20190930'
#outdir = '../output/experiment3_final_abrupt_20191001'
#outdir = '../output/experiment3_final_abrupt_20191002'
outdir = '../output/experiment3_final_abrupt_20191002_v2'
if not os.path.exists(outdir):
    os.makedirs(outdir)


"""
def generate_data_experiment3(length1=500, length2=600, n_repeat=50,
                              y_max=1.0, y_max2=0.9, sigma=0.1, seed=123):
    np.random.seed(seed)

    X_list = []

    # SEGMENT1
    for i in range(n_repeat):
        X1 = sigma * np.random.randn(length1)
        X_list.append(X1)

        X2 = y_max + sigma * np.random.randn(length1)
        X_list.append(X2)

    # SEGMENT2
    X3 = sigma * np.random.randn(length2)
    X_list.append(X3)
    X4 = y_max2 + sigma * np.random.randn(length2)
    X_list.append(X4)
    X5 = sigma * np.random.randn(length2)
    X_list.append(X5)
    X6 = y_max2 + sigma * np.random.randn(length2)
    X_list.append(X6)

    X = np.hstack(X_list)
    X = X.reshape(-1, 1)
    return X
"""

def generate_data_experiment3(length1=500, length2=600, n_repeat=50,
                              y_max=1.0, y_max2=0.9, sigma=0.1, seed=123):
    np.random.seed(seed)

    X_list = []
    real_changepoints = []
    total_length = 0

    # SEGMENT1
    for i in range(n_repeat):
        X1 = sigma * np.random.randn(length1)
        X_list.append(X1)
        total_length += len(X1)
        real_changepoints.append(total_length)

        X2 = y_max + sigma * np.random.randn(length1)
        total_length += len(X2)
        X_list.append(X2)
        real_changepoints.append(total_length)

    # SEGMENT2
    X3 = sigma * np.random.randn(length2)
    total_length += len(X3)
    X_list.append(X3)
    real_changepoints.append(total_length)
    metapoint = total_length
    
    X4 = y_max2 + sigma * np.random.randn(length2*3)
    total_length += len(X3)
    X_list.append(X4)
    real_changepoints.append(total_length)
    #X5 = sigma * np.random.randn(length2)
    #X_list.append(X5)
    #X6 = y_max2 + sigma * np.random.randn(length2)
    #X_list.append(X6)

    X = np.hstack(X_list)
    X = X.reshape(-1, 1)
    return X, real_changepoints, metapoint



def generate_data_experiment3_gradual(length1=500, length2=1000, n_repeat=50, 
                              y_max=1.0, y_max2=0.9, sigma=0.1, seed=123):
    np.random.seed(seed)
    
    X_list = []
    real_changepoints = []
    
    total_length = 0 
    
    # SEGMENT1
    for i in range(n_repeat):
        len1_frac = np.random.randint(-int(0.1*length1), int(0.1*length1))
        X1 = sigma * np.random.randn(length1 + len1_frac)
        X_list.append(X1)
        total_length += len(X1)
        real_changepoints.append(total_length)
        
        len2_frac = np.random.randint(-int(0.1*length1), int(0.1*length1))
        X2 = y_max * np.arange(1, length1 + len2_frac + 1)/ (length1 + len2_frac) + sigma * np.random.randn(length1 + len2_frac)
        X_list.append(X2)
        total_length += len(X2)
        real_changepoints.append(total_length)
        
        len3_frac = np.random.randint(-int(0.1*length1), int(0.1*length1))
        X3 = y_max + sigma * np.random.randn(length1 + len3_frac)
        X_list.append(X3)
        total_length += len(X3)
        real_changepoints.append(total_length)
        
        len4_frac = np.random.randint(-int(0.1*length1), int(0.1*length1))
        X4 = y_max - y_max * np.arange(1, length1 + len4_frac + 1)/(length1 + len4_frac) + sigma * np.random.randn(length1 + len4_frac)
        X_list.append(X4)
        total_length += len(X4)
        real_changepoints.append(total_length)
    
    metapoint = len(X_list)

    # SEGMENT2
    len5_frac = np.random.randint(-int(0.1*length2), int(0.1*length2))
    X5 = sigma * np.random.randn(length2 + len5_frac)
    X_list.append(X5)
    total_length += len(X5)
    real_changepoints.append(total_length)
    
    len6_frac = np.random.randint(-int(0.1*length2), int(0.1*length2))
    X6 = y_max2 + sigma * np.random.randn(length2 + len6_frac)
    X_list.append(X6)
    total_length += len(X6)
    real_changepoints.append(total_length)
    
    len7_frac = np.random.randint(-int(0.1*length2), int(0.1*length2))
    X7 = sigma * np.random.randn(length2 + len7_frac)
    X_list.append(X7)
    total_length += len(X7)
    real_changepoints.append(total_length)
    
    len8_frac = np.random.randint(-int(0.1*length2), int(0.1*length2))
    X8 = y_max2 + sigma * np.random.randn(length2 + len8_frac)
    X_list.append(X8)
    total_length += len(X8)
    real_changepoints.append(total_length)

    X = np.hstack(X_list)
    X = X.reshape(-1, 1)
    
    return X, real_changepoints, metapoint


def detect_change_points_mdl(mdl_stats):
    mdl_stats = np.array(mdl_stats)
    idxes_stats_positive = np.where(mdl_stats > 0)[0]

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
        cp = s + np.argmax(mdl_stats[s:e + 1])
        change_points.append(cp)

    change_points = np.array(change_points)
    return change_points


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


def calc_auc(fars, benefits):
    fars = np.array(fars)
    benefits = np.array(benefits)
    # sort by ascending order
    # idx_ordered = np.argsort(fars)
    idx_ordered = np.lexsort((fars, benefits))
    fars_ordered = fars[idx_ordered]
    benefits_ordered = benefits[idx_ordered]
    # if abs(fars_ordered[0]) > 1e-6:
    # if fars_ordered[0] != 0.0:
    if np.abs(fars_ordered[0]) > 1e-6:
        # fars_ordered = np.hstack((0, 0, fars_ordered))
        fars_ordered = np.hstack((0, 0, fars_ordered))
        # benefits_ordered = np.hstack((0, benefits_ordered[0], benefits_ordered))
        # fars_ordered = np.hstack((0, fars_ordered[0], fars_ordered))
        # benefits_ordered = np.hstack((0, 0, benefits_ordered))
        benefits_ordered = np.hstack((0, benefits_ordered[0], benefits_ordered))
        # fars_ordered = np.hstack((0, fars_ordered))
        # benefits_ordered = np.hstack((0, benefits_ordered))
    elif benefits_ordered[0] != 0.0:
        fars_ordered = np.hstack((0, fars_ordered))
        benefits_ordered = np.hstack((0, benefits_ordered))

    # calculate AUC
    # auc = np.abs(np.sum(np.diff(fars_ordered/np.max(fars_ordered)) *
    #                       np.abs(benefits_ordered[:-1])/np.max(np.abs(benefits_ordered[:-1])))
    # )
    if np.all(benefits_ordered == 0):
        auc = 0.0
    else:
        auc = np.trapz(benefits_ordered,
                       fars_ordered / np.max(fars_ordered))
    return auc


def calc_benefit_far_state(change_points, stats, metapoint, length, n_repeat, limit=5):
    benefits = []
    fars = []

    thresholds = np.sort(stats[~np.isnan(stats)]) - 1e-3
    thresholds = np.linspace(thresholds[0], thresholds[-1], 100)

    for thr in thresholds:
        idxes_over_thr = np.where(stats >= thr)[0]
        within_tol_interval = np.logical_and(
            0 <= (change_points[idxes_over_thr] - metapoint),
            (change_points[idxes_over_thr] - metapoint) <= limit * length)
        # benefit
        benefit = 0.0
        if np.any(within_tol_interval):
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

    return benefits, fars


from scipy.special import gammaln


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

### Change detection
def gridsearch_bocpd(
        X, 
        length1, length2, 
        real_changepoints,
        LAMBDA,
        THRESHOLD, 
        ALPHA=0.1,
        BETA=1.,
        KAPPA=1.,
        MU=0.,
        DELAY=15,
        n_repeat=10,
        ymax=1.0,
        delay=100,
        delay_cp=30, 
        n_trial=10, 
        seed=0
):
    aucs = []
    T = len(X)

    scores = []
    # BOCPD
    bocd = BOCD(partial(constant_hazard, LAMBDA),
                StudentT(ALPHA, BETA, KAPPA, MU), X)
    change_points = []
    scores = []
    for x in X[:DELAY, 0]:
        bocd.update(x)
    for x in tqdm.tqdm(X[DELAY:, 0]):
        bocd.update(x)
        if bocd.growth_probs[DELAY] >= THRESHOLD:
            change_points.append(bocd.t - DELAY + 1)
        score = np.sum(bocd.growth_probs[:bocd.t - DELAY] * 1.0 / (1.0 + np.arange(1, bocd.t - DELAY + 1)))
        scores.append(score)

    change_points = np.array(change_points)
    scores = np.array(scores)

    # AUCs for change points

    thresholds = np.sort(scores[~np.isnan(scores)]) - 1e-3
    thresholds = np.linspace(thresholds[0], thresholds[-1], 100)

    """
    fars, benefits = [], []
    for thr in thresholds:
        idxes_over_thr = np.where(scores >= thr)[0]
        n_fp = len(idxes_over_thr)

        benefit_thr = []
        n_fp_thr = []
        for r_cp in real_changepoints:
            benefit = 0.0
            ok = np.logical_and(idxes_over_thr - r_cp >= 0,
                                idxes_over_thr - r_cp <= delay)
            if any(ok):
                benefit = 1 - (idxes_over_thr[ok][0] - r_cp) / delay

            n_fp = np.sum(
                np.logical_or(
                    idxes_over_thr < r_cp,
                    idxes_over_thr >= r_cp + delay
                )
            )
 
            benefit_thr.append(benefit)
            n_fp_thr.append(n_fp)

        benefits.append(benefit_thr)
        fars.append(n_fp_thr)
        
    benefits = np.array(benefits).T
    fars = np.array(fars).T

    aucs_i = []
    for j in range(fars.shape[0]):
        benefit = benefits[j, :]
        far = fars[j, :]
        auc = calc_auc(far, benefit)
        aucs_i.append(auc)

    aucs.append(np.mean(aucs_i))

    return np.array(aucs), change_points
    """
    
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
                benefit += 1 - (idxes_over_thr[ok][0] - r_cp)/delay
                count += 1
        #if count >= 1:
        #    benefit /= count
            
        fars.append(n_fp)
        benefits.append(benefit)
        
    auc = calc_auc(np.array(fars), np.array(benefits)/np.max(benefits))
    
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
    
    diff = change_points - 2*n_repeat*length1 - length2
    idxes = np.where(diff >= 0)[0]
    d = diff[idxes[0]]
    
    #return auc, change_points
    return auc, change_points, precision, recall, d

    """
    for i in tqdm.tqdm(range(n_trial)):
        # generate data
        #X = generate_data_experiment3(length1, length2, n_repeat=n_repeat,
        #                              y_max=ymax,
        #                              seed=seed)
        T = len(X)
        # SMDL
        # smdl = SMDL(w, T, Norm1D, 0.05)
        # mdl_stats = [np.nan] * w + \
        #            [smdl.calc_change_score(X[(i-w):(i+w), :].ravel(),
        #            mu_max=2.0, sigma_min=0.005) \
        #            for i in range(w, T-w)] + \
        #            [np.nan] * w
        # mdl_stats = np.array(mdl_stats)

        # detect change points
        # change_points = detect_change_points_mdl(mdl_stats - epsilon)

        scores = []
        # BOCPD
        bocd = BOCD(partial(constant_hazard, LAMBDA),
                    StudentT(ALPHA, BETA, KAPPA, MU), X)
        change_points = []
        scores = []
        for x in X[:DELAY, 0]:
            bocd.update(x)
        for x in tqdm.tqdm(X[DELAY:, 0]):
            bocd.update(x)
            if bocd.growth_probs[DELAY] >= THRESHOLD:
                change_points.append(bocd.t - DELAY + 1)
            score = np.sum(bocd.growth_probs[:bocd.t - DELAY] * 1.0 / (1.0 + np.arange(1, bocd.t - DELAY + 1)))
            scores.append(score)

        change_points = np.array(change_points)
        scores = np.array(scores)

        # AUCs for change points

        thresholds = np.sort(scores[~np.isnan(scores)]) - 1e-3
        thresholds = np.linspace(thresholds[0], thresholds[-1], 100)

        fars, benefits = [], []
        for thr in thresholds:
            idxes_over_thr = np.where(scores >= thr)[0]
            n_fp = len(idxes_over_thr)

            benefit_thr = []
            n_fp_thr = []
            for r_cp in real_changepoints:
                benefit = 0.0
                ok = np.logical_and(idxes_over_thr - r_cp >= 0,
                                    idxes_over_thr - r_cp <= delay)
                if any(ok):
                    benefit = 1 - (idxes_over_thr[ok][0] - r_cp) / delay

                n_fp = np.sum(
                    np.logical_or(
                        idxes_over_thr < r_cp,
                        idxes_over_thr >= r_cp + delay
                    )
                )
 
                benefit_thr.append(benefit)
                n_fp_thr.append(n_fp)
           
            benefits.append(benefit_thr)
            fars.append(n_fp_thr)
        
        benefits = np.array(benefits).T
        fars = np.array(fars).T

        aucs_i = []
        for j in range(fars.shape[0]):
            benefit = benefits[j, :]
            far = fars[j, :]
            auc = calc_auc(far, benefit)
            aucs_i.append(auc)

        aucs.append(np.mean(aucs_i))

    return np.array(aucs), change_points
    """

    """
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
        if count >= 1:
            benefit /= count

        fars.append(n_fp)
        benefits.append(benefit)

    auc = calc_auc(np.array(fars), np.array(benefits))

    return auc, change_points, X
    """


### Metachange detection along time
def calc_metachange_time_md(X, change_points, r=0.05):
    change_points_diff = np.diff(change_points)

    lambdas_hat = np.array(
        [(1 - (1 - r) ** (i + 1)) / \
         (r * np.sum((1 - r) ** np.arange(i, -1, -1) * change_points_diff[:i + 1])) \
         for i in range(len(change_points_diff))]
    )
    codelen = -np.log(lambdas_hat[:-1]) + lambdas_hat[:-1] * change_points_diff[1:]
    #change_rate_codelen = np.diff(codelen) / codelen[:-1]

    return codelen


def calc_metachange_time_vd(X, change_points, seed=0):
    change_points_diff = np.diff(change_points)

    vdetector = VolatilityDetector(seed=seed)
    relvol = np.array([vdetector.detect(cpd) for cpd in change_points_diff])

    return relvol


def calc_benefit_far_mcat(change_rate_codelen, change_points,
                          metapoint, 
                         length1, length2, n_repeat,
                         limit=5, r=0.2, B=32, R=32, seed=0):
    ## MCAT
    benefits_md_i, fars_md_i = [], []
    thr_list = np.sort(np.abs(change_rate_codelen[B + R - 3:]))
    thr_list = np.linspace(thr_list[0], thr_list[-1], 100)

    #metapoint = 2*n_repeat*length1 + length2

    for thr in thr_list:
        # metachange detector
        idxes_over_thr_after_cp_within = np.where(
            np.logical_and(
                np.logical_and(
                    #change_points[B + R:] >= 2 * length1 * n_repeat + length2,  # + blockSize,
                    #change_points[B + R:] < 2 * length1 * n_repeat + length2 + limit * length2
                    change_points[B + R:] >= metapoint,  # + blockSize,
                    change_points[B + R:] < metapoint + limit * length2
                ),
                np.abs(change_rate_codelen[B + R - 3:]) > thr
            ))[0]
        if len(idxes_over_thr_after_cp_within) == 0:
            benefit = 0.0
        else:
            #benefit = 1 - (change_points[B + R + idxes_over_thr_after_cp_within[0]] - 2 * length1 * n_repeat - length2) / (
            benefit = 1 - (change_points[B + R + idxes_over_thr_after_cp_within[0]] - metapoint) / (
            limit * length2)

        # false positive rate
        n_fp = np.sum(
            np.logical_and(
                np.logical_or(
                    #change_points[B + R:] >= 2 * length1 * n_repeat + length2 + limit * length2,
                    #change_points[B + R:] < 2 * length1 * n_repeat + length2
                    change_points[B + R:] >= metapoint + limit * length2,
                    change_points[B + R:] < metapoint
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


def calc_benefit_far_vd(change_points, change_points_diff,
                        metapoint,
                       length1, length2, n_repeat, limit=5,
                       B=32, R=32, 
                       blockSize=32, seed=0):
    cps = change_points[1:]

    #metapoint = 2*n_repeat*length1 + length2

    ## volatility detector
    vdetector = VolatilityDetector(seed=seed, b=B, r=R)
    relvol = np.array([vdetector.detect(cpd) for cpd in change_points_diff])

    benefits_vd_i, fars_vd_i = [], []
    bnd_list = np.sort(np.abs(relvol[~np.isnan(relvol)] - 1.0)) - 1e-3
    #bnd_list = np.linspace(bnd_list[0], bnd_list[-1], 100)
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
            #cps_over_thr - 2 * n_repeat * length1 - length2 >= blockSize,
            #cps_over_thr - 2 * n_repeat * length1 - length2 < limit * length2)
            cps_over_thr - metapoint >= 0,
            cps_over_thr - metapoint < limit * length2)

        # benefit
        benefit = 0.0
        if np.any(within_tol_interval):
            #dist_from_cp = np.abs(cps_over_thr[within_tol_interval] - 2 * n_repeat * length1 - length2)
            dist_from_cp = np.abs(cps_over_thr[within_tol_interval] - metapoint)
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
                    #cps < 2 * n_repeat * length1 + length2,
                    #cps >= 2 * n_repeat * length1 + length2 + limit * length2
                    cps < metapoint,
                    cps >= metapoint + limit * length2
                )
            )
        )
        fars_vd_i.append(n_fp)

    return benefits_vd_i, fars_vd_i, relvol


def gridsearch_mct(
         length1, length2,
         change_points_diff,
         change_points,
         metapoint, 
         r=0.2,
         B=32, R=32,
         n_repeat=10,
         limit=1,
         n_trial=20, 
         seed=0
):
    lambdas_hat = np.array(
        [(1 - (1 - r) ** (i + 1)) / \
         (r * np.sum((1 - r) ** np.arange(i, -1, -1) * change_points_diff[:i + 1])) \
         for i in range(len(change_points_diff))]
    )
    codelen = -np.log(lambdas_hat[:-1]) + lambdas_hat[:-1] * change_points_diff[1:]
    change_rate_codelen = np.diff(codelen) / codelen[:-1]

    benefits_md, fars_md = calc_benefit_far_mcat(
        change_rate_codelen, change_points,
        metapoint, 
        length1, length2, n_repeat,
        limit=limit, r=r, B=B, R=R, seed=seed)
    auc_md = calc_auc(np.array(fars_md), np.array(benefits_md))

    #print('## volatility detector')
    benefits_vd, fars_vd, relvol = calc_benefit_far_vd(
        change_points, change_points_diff,
        metapoint, 
        length1, length2, n_repeat, limit=limit, 
        B=B, R=R, seed=seed
    )
    auc_vd = calc_auc(np.array(fars_vd), np.array(benefits_vd))

    return auc_md, auc_vd, codelen, relvol


### Metachange detection along state
def gridsearch_mcs(
         X, 
         length1, length2, 
         change_points,
         metapoint,
         h=50,
         n_repeat=10,
         limit=1
):
    # AUC for metachange
    mcas = calc_metachange_stats_v2(X, change_points, h=h)
    #metapoint = 2*n_repeat*length1 + length2
    benefits_mcas_i, fars_mcas_i = calc_benefit_far_state(
                                       change_points[1:], mcas,
                                       metapoint, length2,
                                       n_repeat, limit=limit)
    auc_mcas = calc_auc(np.array(fars_mcas_i),
                        np.array(benefits_mcas_i))

    #print('## SMDL-MC')
    T = len(X)
    #smdl = SMDL(h, T, Norm1D, 0.05)
    smdl = SMDL(Norm1D, 0.05)
    mdl_stats = [np.nan] * h + \
                [smdl.calc_change_score(X[(i-h):(i+h), :].ravel(), h, 
                mu_max=2.0, sigma_min=0.005) \
                for i in range(h, T-h)] + \
                [np.nan] * h
    mdl_stats = np.array(mdl_stats)

    mdl_stats_rate = np.abs(np.diff(mdl_stats[change_points])/
                                    mdl_stats[change_points][:-1])
    benefits_smdlmc, fars_smdlmc = calc_benefit_far_state(
                                           change_points[1:], mdl_stats_rate,
                                           metapoint, length2,
                                           n_repeat, limit=limit)

    auc_smdlmc = calc_auc(np.array(fars_smdlmc),
                          np.array(benefits_smdlmc))

    return auc_mcas, auc_smdlmc, mcas



results = []

n_repeat = 50
n_trial = 20
limit = 1

r_list = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
#h_list = np.array([50, 100, 150])
R_list = np.array([16, 24, 32, 40])
lambda_list = np.array([1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0])

#for (length1, length2) in permutations([400, 450, 500, 600], 2):
#for (length1, length2) in permutations([400, 500, 600], 2):
for (length1, length2) in permutations([400, 450, 500], 2):
#for (length1, length2) in permutations([500, 1000, 1500, 2000], 2):
    #real_changepoints = np.linspace(length1, 2*n_repeat*length1 + 3*length2, length2)
    #real_changepoints = np.hstack((np.arange(length1, 2*n_repeat*length1 + length1, 
    #                                         length1),
    #                               np.arange(2*n_repeat*length1 + length2, 
    #                                         2*n_repeat*length1 + 4*length2, length2)))
    #h = 100
    h_list = np.array([int(0.1*length1), int(0.2*length1)])
    seed = 0
    for i in tqdm.tqdm(range(n_trial)):
        #X, real_changepoints, metapoint = generate_data_experiment3_gradual(length1, length2, n_repeat=n_repeat, seed=i)
        #X, real_changepoints, metapoint = generate_data_experiment3(length1, length2, n_repeat=n_repeat, seed=i)
        #X, real_changepoints, metapoint = generate_data_experiment3(length1, length2, n_repeat=n_repeat, y_max=0.5, y_max2=0.4, seed=i)
        X, real_changepoints, metapoint = generate_data_experiment3(length1, length2, n_repeat=n_repeat, y_max=0.5, y_max2=0.45, seed=i)
        for LAMBDA in [100, 300, 600]:
            for THRESHOLD in [0.1, 0.3]:
                auc_cp, change_points, precision, recall, delay = gridsearch_bocpd(
                             X, 
                             length1, length2,
                             real_changepoints,
                             LAMBDA=LAMBDA, THRESHOLD=THRESHOLD,
                             n_repeat=n_repeat,
                             n_trial=5,
                             seed=seed)
                
                print(change_points)

                change_points_diff = np.diff(change_points)

                print('# Metachange detection along time')
                seed_t = 0
                for r in r_list:
                    for R in R_list:
                        try:
                            auc_mct, auc_vd, codelen, relvol = gridsearch_mct(
                                length1, length2,
                                change_points_diff,
                                change_points,
                                metapoint, 
                                r=r,
                                B=R, R=R,
                                n_repeat=n_repeat,
                                limit=limit, 
                                seed=seed_t
                            )
                        except:
                            auc_mct = np.nan
                            auc_vd = np.nan
                            codelen = np.nan
                            relvol = np.nan

                        seed_t += 1

                        print('# Metachange detection along state')
                        for h in h_list:
                            try:
                                auc_mcas, auc_smdlmc, mcas = gridsearch_mcs(
                                    X, 
                                    length1, length2, 
                                    change_points,
                                    metapoint, 
                                    h=h, 
                                    n_repeat=n_repeat
                                )
                            except:
                                auc_mcas = np.nan
                                auc_smdlmc = np.nan
                                mcas = np.nan
                
                            print('# Metachange detection along time and state')
                            for lam in lambda_list:
                                codelen_integrated = codelen + lam * mcas[1:]
                                benefits_integrated, fars_integrated = \
                                        calc_benefit_far_state(change_points[2:], codelen_integrated,
                                               #metapoint=2*n_repeat*length1+length2,
                                           metapoint=metapoint,
                                           length=length2, n_repeat=n_repeat, limit=limit)
                                auc_integrated = calc_auc(fars_integrated, benefits_integrated)

                                result = pd.DataFrame({
                                   'length1': length1,
                                   'length2': length2,
                                   'LAMBDA': LAMBDA,
                                   'THRESHOLD': THRESHOLD,
                                   'r': r,
                                   'R': R,
                                   'h': h, 
                                   'lam': lam,
                                   'AUC_CP': auc_cp,
                                   'PRECISION_CP': precision,
                                   'RECALL_CP': recall,
                                   'DELAY': delay, 
                                   'AUC_MCAT': auc_mct,
                                   'AUC_VD': auc_vd,
                                   'AUC_MCAS': auc_mcas,
                                   'AUC_SMDLMC': auc_smdlmc,
                                   'AUC_INTEGRATED': [auc_integrated]},
                                   columns=['length1', 'length2',
                                            'LAMBDA', 'THRESHOLD', 'r', 'R', 'h', 'lam',
                                            'AUC_CP', 
                                            'PRECISION_CP', 'RECALL_CP', 'DELAY', 
                                            'AUC_MCAT', 'AUC_VD',
                                            'AUC_MCAS', 'AUC_SMDLMC',
                                            'AUC_INTEGRATED'])
                                results.append(result)
                                print(
                                  'length1:', length1,
                                  'length2:', length2,
                                  'LAMBDA:', LAMBDA,
                                  'THRESHOLD:', THRESHOLD, 
                                  'r:', r,
                                  'R:', R, 
                                  'lam:', lam,
                                  'AUC_CP:', auc_cp,
                                  'PRECISION_CP:', precision, 
                                  'RECALL_CP:', recall, 
                                  'DELAY:', delay, 
                                  'AUC_MCAT:', auc_mct,
                                  'AUC_VD:', auc_vd,
                                  'AUC_MCAS:', auc_mcas,
                                  'AUC_SMDLMC:', auc_smdlmc,
                                  'AUC_INTEGRATED:', auc_integrated
                                )
                        results_df = pd.concat(results)
                        results_df.to_csv(os.path.join(outdir, 'bocpd_auc_mixed.csv'), index=None)

                seed += 1

