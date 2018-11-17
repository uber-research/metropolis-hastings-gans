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
from collections import OrderedDict
import os.path
import numpy as np
import pandas as pd
import scipy.stats as ss
import classification as cl
from colors import calib_2_color
from contrib.dcgan.dcgan import BASE_D
from matplotlib import rcParams, use
use('Agg')  # Allows plotting on non-GUI environments
import matplotlib.pyplot as plt  # noqa E402: mpl requires import after use

ref_method = BASE_D + '_raw'
MAX_ITER = 60
MAX_Z = np.inf


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


opt = get_opts()

# Note this will put type-3 font BS in the pdfs, if it matters
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['axes.labelsize'] = 10
rcParams['legend.fontsize'] = 6


def dict_subset(D, k):
    D = {k_: D[k_] for k_ in k}
    return D


CALIB_CURVES = OrderedDict([('none', BASE_D + '_raw'),
                            ('isotonic', BASE_D + '_iso')])

agg_perf = {}
Z = {}
for ii in xrange(MAX_ITER):
    # Assuming in current dir for now!
    fname = os.path.join(opt.input, '%d_scores.csv' % ii)

    try:
        scores = pd.read_csv(fname, header=0, index_col=False)
    except Exception as e:
        print str(e)
        print 'seem to have hit last file at:'
        print fname
        break

    print 'loaded:'
    print fname

    y_true = scores.pop('label').values
    Z[ii] = cl.calibration_diagnostic(scores, y_true)
Z = pd.DataFrame(Z).T

plt.figure(figsize=(2.75, 2.5))
for label in CALIB_CURVES:
    method = CALIB_CURVES[label]
    S = np.clip(Z[method], -MAX_Z, MAX_Z)
    S = S[:MAX_ITER]
    plt.plot(S.index.values, S.values, '-', label=label, zorder=3,
             color=calib_2_color[label])
plt.plot(S.index, np.full(len(S), ss.norm.ppf(0.025)), 'k--', zorder=2)
plt.plot(S.index, np.full(len(S), ss.norm.ppf(1.0 - 0.025)), 'k--', zorder=2)
plt.plot(S.index, np.full(len(S), ss.norm.ppf(0.025 / MAX_ITER)),
         'k:', zorder=2)
plt.plot(S.index, np.full(len(S), ss.norm.ppf(1.0 - 0.025 / MAX_ITER)),
         'k:', zorder=2)
plt.xlim((0, MAX_ITER - 1))
plt.ylim((-20, 20))
plt.legend(title='calibration')
plt.grid(False)
plt.xlabel('epoch')
plt.ylabel('$Z$')
plt.tight_layout(pad=0)
plt.savefig(os.path.join(opt.output, 'disc_calib.pdf'),
            dpi=300, facecolor='w', edgecolor='w', format='pdf',
            transparent=False, bbox_inches='tight', pad_inches=0)
