import numpy as np
#from numpy.lib.stride_tricks import as_strided
#import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gammaln

from itertools import permutations

import os

from py4j.java_gateway import JavaGateway
import subprocess
import time

import sys

from volatility_detector import VolatilityDetector

import scaw2 as SCAW2

from smdl import SMDL
from model import Bernoulli

outdir = '../output/experiment1_20181109'
if not os.path.exists(outdir):
    os.makedirs(outdir)

np.random.seed(123)

#cmd
beta = 0.05
#detector =
delta = 0.05
blocksize = 32
epsilon_prime = 0.0075
#alpha = 0.6
alpha = 0.6
decaymode = 1
compression_term = 75

class_path = './ICDMFiles_SEED_py4j/'
sys.path.insert(0, class_path)

java_file = 'SEEDChangeDetector'
class_path = '-cp ' + class_path
cmd = "java {0} {1} {2} {3} {4} {5} {6}".format(
            class_path, java_file,
            delta, blocksize, epsilon_prime,
            alpha, compression_term)

p = subprocess.Popen(cmd, shell=True)
time.sleep(3)
gateway = JavaGateway(start_callback_server=True)

detector_app = gateway.entry_point

delta = 0.05
blockSize = 32
epsilonPrime = 0.0075
alpha = 0.6
term = 75
mu = 0.01
length = 100000
n_repeat = 5000
seed = 0
fpr = detector_app.expr_seed_fpr(
      delta, blockSize,
      epsilonPrime, alpha,
      term,
      mu, length,
      n_repeat, seed
)


def experiment1_benefit_false_alarm_rate(
        length1, length2, cmd, detector,
        r=0.5,
        delta=0.05, blocksize=32, epsilon_prime=0.0075,
        alpha=0.6, decaymode=1, compression_term=75,
        B=32, R=32,
        mu1=0.2, mu2=0.8,
        n_repeat=50, n_trial=100,
        # beta_list=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        limit=5
):
    # volatility detector
    benefits_vd = []  # benefit
    fars_vd = []  # false alarm rate

    # metachange detector
    benefits_md = []  # benefit
    fars_md = []  # false alarm rate

    for i in range(n_trial):
        seed = i
        # change detection with SEED
        change_points = detector_app.expr_volshift(
            mu1, mu2, length1, length2, n_repeat, seed)
        change_points_npa = np.array(list(change_points))
        change_points_npa_diff = np.diff(change_points_npa)
        cps = change_points_npa[1:]

        # metachange detector
        lambdas_hat = np.array(
            [(1 - (1 - r) ** (i + 1)) / \
             (r * np.sum((1 - r) ** np.arange(i, -1, -1) * change_points_npa_diff[:i + 1])) \
             for i in range(len(change_points_npa_diff))]
        )
        codelen = -np.log(lambdas_hat[:-1]) + lambdas_hat[:-1] * change_points_npa_diff[1:]
        change_rate_codelen = np.diff(codelen) / codelen[:-1]

        benefits_vd_i = []
        fars_vd_i = []
        benefits_md_i = []
        fars_md_i = []

        # volatility detector
        for thr in np.arange(0.0, 1.001, 0.001):
            try:
                vdetector = VolatilityDetector(beta=thr, seed=i)
                res_vd = [vdetector.detect(change_points_npa_diff[i]) for i in range(len(change_points_npa_diff))]
                is_metachange = np.array([r[0] for r in res_vd])
                relvol = np.array([r[1] for r in res_vd])

                idxes_over_thr = np.where(
                    np.logical_and(
                        relvol >= thr,
                        ~np.isnan(relvol)
                    )
                )[0]
                cps_over_thr = cps[idxes_over_thr]
                within_tol_interval = np.abs(cps_over_thr - 2 * n_repeat * length1) <= limit * length2
                # benefit
                benefit = 0.0
                if np.any(within_tol_interval):
                    dist_from_cp = np.abs(cps_over_thr - 2 * n_repeat * length1)
                    idx = np.argmin(dist_from_cp)
                    if dist_from_cp[idx] <= limit * length2:
                        benefit = 1 - np.abs(cps_over_thr[idx] - 2 * n_repeat * length1) / (limit * length2)
                        benefits_vd_i.append(benefit)
                # false alarm
                far = np.sum(
                    np.logical_and(
                        np.logical_and(
                            relvol >= thr,
                            ~np.isnan(relvol)
                        ),
                        np.logical_or(
                            cps <= 2 * n_repeat * length1 - limit * length2,
                            cps >= 2 * n_repeat * length1 + limit * length2
                        )
                    )
                )
                fars_vd_i.append(far)
            except:
                break

        benefits_vd.append(benefits_vd_i)  # benefit
        fars_vd.append(fars_vd_i)  # false alarm rate

        maxvalue = np.max(np.abs(change_rate_codelen[change_points_npa[3:] >= 2 * length1 * n_repeat + blockSize]))
        for thr in np.arange(0.0, maxvalue, 0.0001):
            # metachange detector
            try:
                # benefit
                idx_first = np.where(np.logical_and(change_points_npa[3:] >= 2 * length1 * n_repeat + blockSize,
                                                    np.abs(change_rate_codelen) >= thr))[0][0] + 3
                benefit = np.max([0, 1 - (change_points_npa[idx_first] - 2 * length1 * n_repeat) / (limit * length2)])
                benefits_md_i.append(benefit)

                # false positive rate
                n_alarm = np.sum(np.abs(change_rate_codelen) >= thr)
                n_fp = np.sum(
                    np.logical_and(
                        np.logical_or(
                            # change_points_npa[3:] >= 2*length1*n_repeat + blockSize + limit*length2,
                            # change_points_npa[3:] < 2*length1*n_repeat + blockSize
                            change_points_npa[B + R:] >= 2 * length1 * n_repeat + blockSize + limit * length2,
                            change_points_npa[B + R:] < 2 * length1 * n_repeat + blockSize
                        ),
                        # np.abs(change_rate_codelen) >= thr
                        np.abs(change_rate_codelen[B + R - 3:]) >= thr
                    )
                )
                fars_md_i.append(n_fp / n_alarm)
            except:
                break

        benefits_md.append(benefits_md_i)
        fars_md.append(fars_md_i)

    return np.array(benefits_vd), np.array(fars_vd), np.array(benefits_md), np.array(fars_md)


for (length1, length2) in permutations([100, 500, 1000, 5000, 10000, 50000, 100000], 2):
    benefits_vd, fars_vd, benefits_md, fars_md = experiment1_benefit_false_alarm_rate(
            length1, length2, cmd, detector_app,
             r=0.5,
             delta=0.05, blocksize=32, epsilon_prime=0.0075,
             alpha=0.6, decaymode=1, compression_term=75,
             mu1=0.2, mu2=0.8,
             n_repeat=50, n_trial=1,
             #beta_list=[0.01, 0.02, 0.03, 0.04, 0.05,
             #                   0.1, 0.15, 0.2, 0.25, 0.3],
             limit=5
    )
    auc_vd = np.abs(np.sum(np.diff(fars_vd.ravel()/np.max(fars_vd.ravel()))*np.abs(benefits_vd.ravel())[:-1]))
    auc_md = np.abs(np.sum(np.diff(fars_md.ravel()/np.max(fars_md.ravel()))*np.abs(benefits_md.ravel())[:-1]))
    print('lengh1=', length1, ' length2=', length2)
    print('  AUC(vd)=', auc_vd, ' AUC(md)=', auc_md)
