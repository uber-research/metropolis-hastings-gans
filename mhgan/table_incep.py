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
import pandas as pd
from benchmark_tools.constants import METHOD, METRIC, STAT
from benchmark_tools.constants import MEAN_COL, ERR_COL, PVAL_COL
import benchmark_tools.sciprint as sp

INCEP = 'incep'
NA = '-'

experiment_dict = {'DCGAN CIFAR-10': ('perf_dcgan_cifar.csv', 17),
                   'WGAN CIFAR-10': ('perf_wgan_cifar.csv', 60),
                   'DCGAN CelebA': ('perf_dcgan_celeba.csv', 17),
                   'WGAN CelebA': ('perf_wgan_celeba.csv', 60)}


def get_opts(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='.',
                        help='folder to find csv dumps from demo script')
    opt = parser.parse_args(args=args)
    return opt


opt = get_opts()

agg_df = []
for exp_name, (exp_file, used_epoch) in experiment_dict.iteritems():
    df = pd.read_csv(os.path.join(opt.input, exp_file), header=[0, 1])

    used = df['epoch', '-'].max() if used_epoch is None else used_epoch
    print 'Using epoch %d for %s' % (used, exp_name)
    df = df[df['epoch', '-'] == used]

    D = OrderedDict([((exp_name, MEAN_COL), df[(INCEP, MEAN_COL)].values),
                     ((exp_name, ERR_COL), df[(INCEP, ERR_COL)].values),
                     ((exp_name, PVAL_COL), df[(INCEP, PVAL_COL)].values)])
    df = pd.DataFrame(data=D, index=df[(METHOD, NA)].values)
    df.columns.names = [METRIC, STAT]

    agg_df.append(df)
perf_df = pd.concat(agg_df, axis=1, join='inner')
perf_df.index.name = METHOD

perf_report = sp.just_format_it(perf_df, shift_mod=3,
                                non_finite_fmt={'nan': '--'}, use_tex=True)
print perf_report
