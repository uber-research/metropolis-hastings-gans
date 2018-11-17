# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.special import logit
from colors import picker_2_color
import mh
from contrib.dcgan.dcgan import BASE_D
from matplotlib import rcParams, use
use('Agg')  # Allows plotting on non-GUI environments
import matplotlib.pyplot as plt  # noqa E402: mpl requires import after use

# Note this will put type-3 font BS in the pdfs, if it matters
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['axes.labelsize'] = 10
rcParams['legend.fontsize'] = 8

fname = '13_scores.csv'
xlim = (-12, 7)

runs = 500
chain_len = 640

method_to_plot = BASE_D + '_iso'
lim = 14

EPSILON = 1e-6
FORCE_GEN = True
LABEL = 'label'


def get_opts(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='.',
                        help='folder to find csv dumps from demo script')
    parser.add_argument('--output', default='.',
                        help='folder to output figures')

    opt = parser.parse_args(args=args)

    # Make the dir
    try:
        os.makedirs(opt.output)
    except OSError:
        # If dir already exists (this is not very robust but what pytorch does)
        pass

    return opt


def kde(x, lower, upper, N=1000):
    x_grid = np.linspace(lower, upper, N)
    k = gaussian_kde(x)
    dd = k(x_grid)
    return x_grid, dd


def base(score, score_max=None):
    idx = len(score) - 1
    return idx, 1.0


def converge_series(disc_f, p_rnd, q_rnd, runs=100, N=1000):
    X = np.zeros((runs, N + 1))
    for rr in xrange(runs):
        s0 = disc_f(p_rnd(1))
        s_fake = disc_f(q_rnd(N))
        s_ = np.concatenate((s0, s_fake))
        w = 1.0 / s_
        X[rr, :] = mh.cumm_mh_sample_distn(s_, w)
    return X


def safe_logit(x):
    y = logit(np.clip(x, EPSILON, 1 - EPSILON))
    return y


opt = get_opts()

np.random.seed(453453)

pickers = {'base': base, 'MH': mh.mh_sample, 'reject': mh.rejection_sample}

scores = pd.read_csv(os.path.join(opt.input, fname), header=0, index_col=False)

y_true = scores.pop(LABEL)
scores_real_df = scores.loc[y_true, :]
scores_fake_df = scores.loc[~y_true, :]

score_delta = pd.DataFrame(data=0.0,
                           index=pickers.keys(), columns=scores.columns,
                           dtype=float)
score_end = {}
for rr in xrange(runs):
    init_idx = np.random.choice(len(scores_real_df))
    for method in scores:
        score_bound = np.max(scores_fake_df[method].values)

        s0 = scores_real_df[method].values[init_idx]
        # Sample with replacement to use empirical distn as approx of true
        # distn on fake scores
        s_fake = np.random.choice(scores_fake_df[method].values,
                                  size=chain_len, replace=True)
        s_ = np.concatenate(([s0], s_fake))
        max_score = np.max(s_)
        max_odds = mh.disc_2_odds_ratio(max_score)
        for picker_name in pickers:
            idx, _ = pickers[picker_name](s_, score_max=score_bound)
            score_ = s_[idx]
            if FORCE_GEN and idx == 0:
                idx, _ = pickers[picker_name](s_[1:], score_max=score_bound)
                score_ = s_[idx + 1]

            score_delta.loc[picker_name, method] += (score_ - max_score)
            L = score_end.setdefault((method, picker_name), [])
            L.append(score_)
score_delta = score_delta / float(runs)

print score_delta.to_string()

plt.figure(figsize=(3, 2.75))
xx, dd = kde(safe_logit(scores_fake_df[method_to_plot].values), -lim, lim)
ax_gan, = plt.plot(xx, dd, label='fake (GAN)', alpha=0.9,
                   color=picker_2_color['base'])
xx, dd = kde(safe_logit(scores_real_df[method_to_plot].values), -lim, lim)
ax_real, = plt.plot(xx, dd, label='real', alpha=0.9,
                    color=picker_2_color['real'])
xx, dd = kde(safe_logit(score_end[(method_to_plot, 'MH')]), -lim, lim)
ax_mhc, = plt.plot(xx, dd, label='MHGAN (cal)', alpha=0.9,
                   color=picker_2_color['MH'])
xx, dd = kde(safe_logit(score_end[(BASE_D + '_raw', 'MH')]), -lim, lim)
ax_mh, = plt.plot(xx, dd, label='MHGAN', alpha=0.9,
                  color=picker_2_color['cherry'])
xx, dd = kde(safe_logit(score_end[(BASE_D + '_raw', 'reject')]), -lim, lim)
ax_drs, = plt.plot(xx, dd, label='DRS', alpha=0.9,
                   color=picker_2_color['reject'])
plt.xlim(xlim)
plt.tick_params(labelleft=False)
plt.xlabel(r"logit $D(\mathbf{x}')$")
plt.ylabel('PDF')
plt.legend(handles=[ax_real, ax_gan, ax_drs, ax_mh, ax_mhc])
plt.grid()
plt.tight_layout(pad=0)
plt.savefig(os.path.join(opt.output, 'score_dist.pdf'),
            dpi=300, facecolor='w', edgecolor='w', format='pdf',
            transparent=False, bbox_inches='tight', pad_inches=0)
