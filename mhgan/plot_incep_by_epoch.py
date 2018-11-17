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

star = 'MH'
mark_epoch = 15
input_fname = 'perf.csv'

name_lookup = {'MH': 'MHGAN', 'base': 'GAN', 'reject': 'DRS'}


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

df = pd.read_csv(os.path.join(opt.input, input_fname), header=[0, 1])

methods_df = df[('method', '-')].str.split('_', expand=True)
methods_df.columns = pd.MultiIndex.from_tuples([('disc', '-'),
                                                ('calib', '-'),
                                                ('picker', '-')])

df = pd.concat((methods_df, df), axis=1)

plt.figure(figsize=(3, 2.75))
idx = (df[('disc', '-')] == BASE_D) & (df[('calib', '-')] == 'raw') & \
    (df[('picker', '-')] == 'MH')
dfm = df[idx]
plt.fill_between(dfm['epoch'].values[:, 0],
                 dfm[('incep', 'mean')].values -
                 dfm[('incep', 'error')].values,
                 dfm[('incep', 'mean')].values +
                 dfm[('incep', 'error')].values,
                 color='k', lw=0, alpha=0.3, zorder=1)
ax_mh, = plt.plot(dfm['epoch'].values, dfm[('incep', 'mean')].values, '-',
                  label='MHGAN', alpha=0.9, zorder=2000,
                  color=picker_2_color['cherry'])

idx = (df[('disc', '-')] == BASE_D) & (df[('calib', '-')] == 'iso') & \
    (df[('picker', '-')] == 'MH')
dfm = df[idx]
plt.fill_between(dfm['epoch'].values[:, 0],
                 dfm[('incep', 'mean')].values -
                 dfm[('incep', 'error')].values,
                 dfm[('incep', 'mean')].values +
                 dfm[('incep', 'error')].values,
                 color='k', lw=0, alpha=0.3, zorder=1)
ax_mhc, = plt.plot(dfm['epoch'].values, dfm[('incep', 'mean')].values, '-',
                   label='MHGAN (cal)', alpha=0.9, zorder=2000,
                   color=picker_2_color['MH'])

idx = (df[('disc', '-')] == BASE_D) & (df[('calib', '-')] == 'raw') & \
    (df[('picker', '-')] == 'reject')
dfm = df[idx]
plt.fill_between(dfm['epoch'].values[:, 0],
                 dfm[('incep', 'mean')].values -
                 dfm[('incep', 'error')].values,
                 dfm[('incep', 'mean')].values +
                 dfm[('incep', 'error')].values,
                 color='k', lw=0, alpha=0.3, zorder=1)
ax_drs, = plt.plot(dfm['epoch'].values, dfm[('incep', 'mean')].values, '-',
                   label='DRS', alpha=0.9, zorder=2000,
                   color=picker_2_color['reject'])

idx = (df[('disc', '-')] == BASE_D) & (df[('calib', '-')] == 'raw') & \
    (df[('picker', '-')] == 'base')
dfm = df[idx]
ax_gan, = plt.plot(dfm['epoch'].values, dfm[('incep', 'mean')].values, '-',
                   label='GAN', alpha=0.9, zorder=2000,
                   color=picker_2_color['base'])

plt.axvline(x=mark_epoch, ymin=0, ymax=1, ls='--', color='k')
plt.xlim((0, 22))
plt.ylim((1.25, 3.55))
plt.legend(handles=[ax_gan, ax_drs, ax_mh, ax_mhc])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('inception score')
plt.tight_layout()
plt.savefig(os.path.join(opt.output, 'per_epoch.pdf'), dpi=300, facecolor='w',
            edgecolor='w', format='pdf', transparent=False,
            bbox_inches='tight', pad_inches=0)
