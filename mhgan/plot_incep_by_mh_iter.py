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
import os.path
import pandas as pd
from colors import picker_2_color
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

fname = 'cumm_incep_15.csv'

INCEP = 'incep'


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

df = pd.read_csv(os.path.join(opt.input, fname), header=[0, 1, 2])
df = df.xs(INCEP, axis=1, level=1)
S = df.mean()
dfm = S.unstack()
dfm.index = dfm.index.astype(int)
dfm = dfm.sort_index()

plt.figure(figsize=(2.75, 2.5))
ax_mh, = plt.plot(dfm.index.values, dfm[BASE_D + '_raw_MH'].values, '.-',
                  label='MHGAN', color=picker_2_color['cherry'])
ax_mhc, = plt.plot(dfm.index.values, dfm[BASE_D + '_iso_MH'].values, '.-',
                   label='MHGAN (cal)', color=picker_2_color['MH'])
ax_gan, = plt.plot(dfm.index.values, dfm[BASE_D + '_raw_base'].values, '.-',
                   label='GAN', color=picker_2_color['base'])
plt.xlim((-0.025 * dfm.index.values[-1], 1.025 * dfm.index.values[-1]))
plt.legend(handles=[ax_gan, ax_mh, ax_mhc])
plt.grid()
plt.xlabel('MC iteration')
plt.ylabel('inception score')
plt.tight_layout(pad=0)
plt.savefig(os.path.join(opt.output, 'plot_per_mh.pdf'),
            dpi=300, facecolor='w', edgecolor='w', format='pdf',
            transparent=False, bbox_inches='tight', pad_inches=0)
