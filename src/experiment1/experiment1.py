%matplotlib inline
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gammaln

from itertools import permutations

import os

from py4j.java_gateway import JavaGateway
import subprocess
import time

import sys

from smdl import SMDL
from model import Bernoulli

from volatility_detector import VolatilityDetector

import tqdm

outdir = '../output/experiment1'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
beta = 0.05
delta = 0.05
blockSize = 32
epsilon_prime = 0.0075
alpha = 0.6
decaymode = 1
compression_term = 75

class_path = './ICDMFiles_SEED_py4j/'
sys.path.insert(0, class_path)

java_file = 'SEEDChangeDetector'
class_path = '-cp ' + class_path
cmd = "java {0} {1} {2} {3} {4} {5} {6}".format(
            class_path, java_file, 
            delta, blockSize, epsilon_prime,
            alpha, compression_term)

p = subprocess.Popen(cmd, shell=True)
time.sleep(3)
gateway = JavaGateway(start_callback_server=True)

detector_app = gateway.entry_point

def experiment1_benefit_false_alarm_rate(
         length1, length2, cmd, detector,
         r=0.5,
         delta=0.05, blocksize=32, epsilon_prime=0.0075,
         alpha=0.6, decaymode=1, compression_term=75,
         B=32, R=32, 
         mu1=0.2, mu2=0.8,
         n_repeat=50, n_trial=100,
         #beta_list=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3], 
         limit=5
):
    # volatility detector
    benefits_vd, fars_vd = [], []   # benefit, false alarm rate
    
    # metachange detector
    benefits_md, fars_md = [], []  # benefit, false alarm rate
    
    for i in tqdm.tqdm(range(n_trial)):
        seed = i
        # change detection with SEED
        change_points = detector_app.expr_volshift(
            mu1, mu2, length1, length2, n_repeat, seed)
        change_points_npa = np.array(list(change_points))
        change_points_npa_diff = np.diff(change_points_npa)
        cps = change_points_npa[1:]

        # metachange detector
        lambdas_hat = np.array(
                [(1 - (1-r)**(i+1)) / \
                 (r * np.sum( (1-r)**np.arange(i, -1, -1) * change_points_npa_diff[:i+1] )) \
                 for i in range(len(change_points_npa_diff))]
        )
        codelen = -np.log(lambdas_hat[:-1]) + lambdas_hat[:-1] * change_points_npa_diff[1:]
        #codelen = -np.log(lambdas_hat) + lambdas_hat * change_points_npa_diff
        change_rate_codelen = np.diff(codelen)/codelen[:-1]
        
        benefits_vd_i, fars_vd_i = [], []
        benefits_md_i, fars_md_i = [], []

        # volatility detector
        #for thr in np.arange(0.0, 0.991, 0.001):
        #    try:
        #vdetector = VolatilityDetector(beta=thr, seed=i)
        vdetector = VolatilityDetector(seed=i)
        #res_vd = [vdetector.detect(change_points_npa_diff[i]) for i in range(len(change_points_npa_diff))]
        relvol = np.array([vdetector.detect(change_points_npa_diff[i]) for i in range(len(change_points_npa_diff))])
        bnd_list = np.sort(np.abs(relvol[~np.isnan(relvol)] - 1.0))
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
                                    cps_over_thr - 2*n_repeat*length1 - blocksize >= 0,
                                    cps_over_thr - 2*n_repeat*length1 <= limit*length2)
            
            # benefit
            benefit = 0.0
            if np.any(within_tol_interval):
                dist_from_cp = np.abs(cps_over_thr[within_tol_interval] - 2*n_repeat*length1)
                    
                benefit = np.sum(1 - (dist_from_cp/(limit*length2)))
            benefits_vd_i.append(benefit)
                
            n_alarm = np.sum(np.logical_and(
                np.logical_or(relvol > 1.0 + bnd, relvol < 1.0 - bnd),
                ~np.isnan(relvol)
            ))
            # false alarm
            n_fp = np.sum(
                np.logical_and(
                    np.logical_and(
                         np.logical_or(relvol > 1.0 + bnd, relvol < 1.0 - bnd),
                        ~np.isnan(relvol)
                    ),
                    np.logical_or(
                        cps <= 2*n_repeat*length1, 
                        cps >= 2*n_repeat*length1 + limit*length2
                    )
                )
            )
            fars_vd_i.append(n_fp)

        benefits_vd.append(np.array(benefits_vd_i))   # benefit
        fars_vd.append(np.array(fars_vd_i))  # false alarm rate
        
        thr_list = np.sort(np.abs(change_rate_codelen[B+R-3:]))
        for thr in thr_list:
            # metachange detector
            # benefit
            idxes_over_thr_after_cp_within = np.where(
                                            np.logical_and(
                                                np.logical_and(
                                                    change_points_npa[B+R:] >= 2*length1*n_repeat + blockSize,
                                                    change_points_npa[B+R:] <= 2*length1*n_repeat + limit*length2
                                                ),
                                                np.abs(change_rate_codelen[B+R-3:]) > thr
                                            ))[0]
            if len(idxes_over_thr_after_cp_within) == 0:
                benefit = 0.0
            else :
                benefit = np.sum(1 - (change_points_npa[B+R+idxes_over_thr_after_cp_within] - 2*length1*n_repeat) / (limit * length2))
            
            # false positive rate
            n_alarm = np.sum(np.abs(change_rate_codelen[B+R-3:]) > thr)
            n_fp = np.sum(
                    np.logical_and(
                        np.logical_or(
                            change_points_npa[B+R:] >= 2*length1*n_repeat + limit*length2, 
                            change_points_npa[B+R:] < 2*length1*n_repeat
                        ), 
                        np.logical_and(
                            np.abs(change_rate_codelen[B+R-3:]) > thr,
                            ~np.isnan(change_rate_codelen[B+R-3:])
                        )
                    )
                 )

            benefits_md_i.append(benefit)
            fars_md_i.append(n_fp)

        benefits_md.append(np.array(benefits_md_i))
        fars_md.append(np.array(fars_md_i))

    return benefits_vd, fars_vd, benefits_md, fars_md

def calc_auc(fars, benefits):
    # sort by ascending order
    idx_ordered = np.lexsort((fars, benefits))
    fars_ordered = fars[idx_ordered]
    benefits_ordered = benefits[idx_ordered]
    if np.abs(fars_ordered[0]) > 1e-6:
        fars_ordered = np.hstack((0, 0, fars_ordered))
        benefits_ordered = np.hstack((0, benefits_ordered[0], benefits_ordered))
    elif benefits_ordered[0] != 0.0:
        fars_ordered = np.hstack((0, fars_ordered))
        benefits_ordered = np.hstack((0, benefits_ordered))

    # calculate AUC
    if np.all(benefits_ordered == 0):
        auc = 0.0
    else:
        auc = np.trapz(benefits_ordered/np.max(benefits_ordered),
                   fars_ordered/np.max(fars_ordered))
    return auc


for (length1, length2) in permutations([500, 1000, 5000, 10000, 50000, 100000], 2):
    benefits_vd, fars_vd, benefits_md, fars_md = experiment1_benefit_false_alarm_rate(
        length1, length2, cmd, detector_app,
        r=0.2, 
        delta=0.05, blocksize=32, epsilon_prime=0.0075,
        alpha=0.6, decaymode=1, compression_term=75,
        mu1=0.2, mu2=0.8,
        n_repeat=50, 
        n_trial=50, 
        limit=5
    )
    
    # AUC (vd)
    auc_vd = []
    for far, benefit in zip(fars_vd, benefits_vd):
        auc_vd.append(calc_auc(far, benefit))
    
    # AUC (md)
    auc_md = []
    for far, benefit in zip(fars_md, benefits_md):
        auc_md.append(calc_auc(far, benefit))
    
    print('length1=', length1, 'length2=', length2)
    print('  AUC(vd):', np.mean(auc_vd), '+/-', np.std(auc_vd))
    print('  AUC(md):', np.mean(auc_md), '+/-', np.std(auc_md))