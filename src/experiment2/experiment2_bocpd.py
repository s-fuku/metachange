import numpy as np
import pandas as pd

import scipy.stats
from scipy.special import gammaln

import os
import tqdm

import pickle

from smdl import SMDL
from model import Norm1D

from mybocpd import BOCD, constant_hazard, StudentT
from functools import partial

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
                              y_max=1.0, y_max2=0.9, sigma=0.1, seed=123):
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

    if idxes_stats_positive[idxes_start[1]-1] not in end:
        end = np.hstack((idxes_stats_positive[idxes_start[1] - 1], end))
    if idxes_stats_positive[-1] not in end:
        end = np.hstack((end, idxes_stats_positive[-1]))

    change_points = []
    for s, e in zip(start, end):
        cp = s + np.argmax(mdl_stats[s:e+1])
        change_points.append(cp)

    change_points = np.array(change_points)
    return change_points

def calc_auc(fars, benefits):
    fars = np.array(fars)
    benefits = np.array(benefits)
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
        

    if np.abs(fars_ordered[-1]) < 1.0 - 1e-6:
        fars_ordered = np.hstack((fars_ordered, 1.0, 1.0))
        benefits_ordered = np.hstack((benefits_ordered, benefits_ordered[-1], 1.0))
        
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

def calc_benefit_far_state(change_points, stats, metapoint, length, n_repeat, limit=5):
    benefits = []
    fars = []
    
    thresholds = np.sort(stats[~np.isnan(stats)]) - 1e-3
    thresholds = np.linspace(thresholds[0], thresholds[-1], 100)
    
    for thr in thresholds:
        idxes_over_thr = np.where(stats >= thr)[0]
        within_tol_interval = np.logical_and(
                     0 <= (change_points[idxes_over_thr] - metapoint), 
                     (change_points[idxes_over_thr] - metapoint) <= limit*length)
        # benefit
        benefit = 0.0
        if np.any(within_tol_interval):
            benefit = 1 - (change_points[idxes_over_thr][within_tol_interval][0] - metapoint)/(limit*length)
        
        # false positive rate
        n_fp = np.sum(
                np.logical_and(
                  np.logical_or(
                    change_points <= metapoint, 
                    change_points >= metapoint + limit*length
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


def calc_metachange_stats_v2(X, change_points, h=100, mu_max=2.0, sigma_min=0.005):
    metachange_stats = []
    
    
    #for t in range(h, len(X)-h):
    for i, cp in enumerate(change_points):
        mean1 = np.mean(X[(cp-h):cp, :].ravel())
        std1 = np.std(X[(cp-h):cp, :].ravel())
        #mean2 = np.mean(X[cp:(cp+h+1), :].ravel())
        #std2 = np.std(X[cp:(cp+h+1), :].ravel())
        mean2 = np.mean(X[(cp+1):(cp+h+1), :].ravel())
        std2 = np.std(X[(cp+1):(cp+h+1), :].ravel())
                
        if i == 0:
            mean1_prev, std1_prev = mean1, std1
            mean2_prev, std2_prev = mean2, std2
            continue
        
        metachange_up = np.mean(-scipy.stats.norm(mean1 + (mean2_prev - mean1_prev), std1 + (std2_prev - std1_prev)).logpdf(X[(cp+1):(cp+h+1), :].ravel()))
        metachange_down = np.mean(-scipy.stats.norm(mean1 - (mean2_prev - mean1_prev), std1 - (std2_prev - std1_prev)).logpdf(X[(cp+1):(cp+h+1), :].ravel()))

        metachange = np.nanmin([metachange_up, metachange_down]) - 0.5 * np.log((16 * np.abs(mu_max))/ (np.pi * sigma_min**2)) \
                     +np.log(2.0) + 1.0 - gammaln(0.5)

        metachange_stats.append(metachange)
        #print(metachange)
        
        mean1_prev, std1_prev = mean1, std1
        mean2_prev, std2_prev = mean2, std2
    
    return np.array(metachange_stats)
    #return np.abs(np.diff(metachange_stats))



def gridsearch_bocpd(
         X, 
         length,
         real_changepoints, 
         LAMBDA=100,
         ALPHA=0.1,
         BETA=1.,
         KAPPA=1.,
         MU=0.,
         DELAY=15,
         THRESHOLD=0.5,
         n_repeat=10,
         ymax=1.0,  
         delay=100, 
         delay_cp=30, 
         seed=1
):
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
    print(change_points)
    scores = np.array(scores)

    # AUCs for change points
        
    thresholds = np.sort(scores[~np.isnan(scores)]) - 1e-3
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
                benefit += 1 - (idxes_over_thr[ok][0] - r_cp)/delay
                count += 1
        #if count >= 1:
        #    benefit /= count
            
        fars.append(n_fp)
        benefits.append(benefit)
        
    auc = calc_auc(np.array(fars)/np.max(fars), np.array(benefits)/np.max(benefits))

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
    
    #return aucs, change_points, precision, recall, d
    return auc, change_points, precision, recall, d, tp, fp, fn
    #return auc, change_points

def gridsearch_mcd(
         X, 
         length,
         change_points, 
         h=50,
         n_repeat=10,
         ymax=1.0,  
         limit=0.5, 
         n_trial=10
): 
    # AUC for metachange
    mcas = calc_metachange_stats_v2(X, change_points, h=h)

    #metapoint = 4*n_repeat*length + length
    metapoint = 2*n_repeat*length
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
                                           metapoint, length,
                                           n_repeat, limit=limit)

    auc_smdlmc = calc_auc(np.array(fars_smdlmc), 
                          np.array(benefits_smdlmc))

    return auc_mcas, auc_smdlmc

results = []

n_repeat = 10
n_trial = 50

ALPHA = 0.1
BETA=1.
KAPPA=1.
MU=0.
DELAY=15

limit = 0.5

for length in [500, 1000, 2000]:
    real_changepoints = np.arange(length, 2*n_repeat*length+2*length, length) 
    for i in tqdm.tqdm(range(n_trial)):
        # generate data
        X = generate_data_experiment2(length, n_repeat=n_repeat, ymax=0.5, seed=i)
        #X = generate_data_experiment2_gradual(length, n_repeat=n_repeat, seed=i)
        for LAMBDA in [100, 300, 600]:
            for THRESHOLD in [0.1, 0.2, 0.3]:
                auc_cp, change_points, precision, recall, delay, tp, fp, fn = gridsearch_bocpd(
                             X, 
                             length,
                             real_changepoints,
                             LAMBDA=LAMBDA, 
                             THRESHOLD=THRESHOLD,
                             n_repeat=n_repeat, 
                             seed=i)
                print(change_points)
                aucs_mcas = []
                aucs_smdlmc = []
                #for h in [int(0.1*length), int(0.2*length), int(0.3*length)]:
                #for h in [50, 100, 150]:
                for h in [int(0.1*length), int(0.2*length)]:
                    try:
                        auc_mcas, auc_smdlmc = gridsearch_mcd(
                            X, 
                            length, 
                            change_points, 
                            h=h, 
                            limit=limit, 
                            n_trial=n_trial,
                            n_repeat=n_repeat
                        ) 
                        aucs_mcas.append(auc_mcas)
                        aucs_smdlmc.append(auc_smdlmc)
                    except:
                        auc_mcas = np.nan
                        auc_smdlmc = np.nan

                    result = pd.DataFrame({
                               'length': length, 
                               'h': h,
                               'LAMBDA': LAMBDA,
                               'THRESHOLD': THRESHOLD,
                               'ALPHA': ALPHA,
                               'BETA': BETA,
                               'KAPPA': KAPPA, 
                               'MU': MU,
                               'DELAY': DELAY, 
                               'AUC_CP': auc_cp,
                               'PRECISION_CP': precision, 
                               'RECALL_CP': recall, 
                               'DELAY_CP': delay,
                               'TP': tp,
                               'FP': fp,
                               'FN': fn, 
                               'AUC_MCAS': aucs_mcas, 
                               'AUC_SMDLMC': aucs_smdlmc}, 
                               columns=['length', 
                                        'h', 
                                        'LAMBDA',
                                        'THRESHOLD',
                                        'ALPHA',
                                        'BETA',
                                        'KAPPA', 
                                        'MU',
                                        'DELAY', 
                                        'AUC_CP', 'PRECISION_CP', 'RECALL_CP', 'DELAY_CP', 
                                        'TP', 'FP', 'FN',
                                        'AUC_MCAS', 'AUC_SMDLMC'])
                    results.append(result)
                    print(
                          'length:', length,
                          'h:', h, 
                          'LAMBDA:', LAMBDA,
                          'THRESHOLD:', THRESHOLD,
                          'ALPHA:', ALPHA,
                          'BETA:', BETA,
                          'KAPPA:', KAPPA, 
                          'MU:', MU,
                          'DELAY_CP:', delay, 
                          'AUC_CP:', auc_cp, 
                          'PRECISION_CP:', precision,
                          'RECALL_CP:', recall, 
                          'TP:', tp, 
                          'FP:', fp,
                          'FN:', fn, 
                          'AUC_MCAS:', auc_mcas,
                          'AUC_SMDLMC:', auc_smdlmc
                    )

                    results_df = pd.concat(results)
                    results_df.to_csv(os.path.join(outdir, 'bocpd_auc_mixed.csv'), index=None)
