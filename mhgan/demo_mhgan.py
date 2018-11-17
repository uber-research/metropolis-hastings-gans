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
from collections import defaultdict
import os.path
from tempfile import mkdtemp
from random import seed as py_seed
import numpy as np
import pandas as pd
from scipy.special import expit as logistic
import torch
import torchvision.utils as vutils
import benchmark_tools.benchmark_tools as bt
import benchmark_tools.classification as btc
import benchmark_tools.sciprint as sp
from benchmark_tools.constants import METHOD, METRIC
from contrib.dcgan.dcgan import gan_trainer, BASE_D
from contrib.dcgan.dcgan_loader import get_data_loader, get_opts
from contrib.inception_score.inception_score import \
    inception_score_precomp, inception_score_fast
import classification as cl
import mh

SEED_MAX = 2**32 - 1  # np.random.randint can't accept seeds above this
LABEL = 'label'
NA_LEVEL = '-'
DBG = False  # Use bogus incep scores for faster debugging
SKIP_INIT_EVAL = True
SAVE_IMAGES = True
INCEP_SCORE = True

INCEP = 'incep'


def const_dict(val):
    D = defaultdict(lambda: val)
    return D


def base(score, score_max=None):
    '''This is a normal GAN. It always just selects the first generated image
    in a series.
    '''
    idx = 0
    return idx, 1.0


PICK_DICT = {'MH': mh.mh_sample, 'base': base, 'reject': mh.rejection_sample}

# ============================================================================
# These functions are here just to illustrate the kind of interface the trainer
# iterator needs to provide.


def gen_dummy(batch_size):
    image_size = 32
    nc = 3

    X = np.random.randn(batch_size, nc, image_size, image_size)
    return X


def gen_and_disc_dummy(batch_size):
    disc_names = (BASE_D,)

    X = gen_dummy(batch_size)
    scores = {k: np.random.rand(X.shape[0]) for k in disc_names}
    return X, scores


def trainer_dummy():
    disc_names = (BASE_D,)

    while True:
        scores_real = {k: np.random.rand(320) for k in disc_names}
        yield gen_dummy, gen_and_disc_dummy, scores_real

# ============================================================================
# Validation routines:


def validate_X(X):
    assert isinstance(X, np.ndarray)
    assert X.dtype.kind == 'f'
    batch_size, nc, image_size, _ = X.shape
    assert X.shape == (batch_size, nc, image_size, image_size)
    assert np.all(np.isfinite(X))
    return X


def validate_scores(scores):
    assert isinstance(scores, dict)
    for sv in scores.values():
        assert isinstance(sv, np.ndarray)
        assert sv.dtype.kind == 'f'
        assert sv.ndim == 1
        assert np.all(0 <= sv) and np.all(sv <= 1)
    scores = pd.DataFrame(scores)
    return scores


def validate(R):
    '''
    X : ndarray, shape (batch_size, nc, image_size, image_size)
    scores : dict of str -> ndarray of shape (batch_size,)
    '''
    X, scores = R
    X = validate_X(X)
    scores = validate_scores(scores)
    assert len(X) == len(scores)
    return X, scores

# ============================================================================


def batched_gen_and_disc(gen_and_disc, n_batches, batch_size):
    '''
    Get a large batch of images. Pytorch might run out of memory if we set
    the batch size to n_images=n_batches*batch_size directly.

    g_d_f : callable returning (X, scores) compliant with `validate`
    n_images : int
        assumed to be multiple of batch size
    '''
    X, scores = zip(*[validate(gen_and_disc(batch_size))
                      for _ in xrange(n_batches)])
    X = np.concatenate(X, axis=0)
    scores = pd.concat(scores, axis=0, ignore_index=True)
    return X, scores


def enhance_samples(scores_df, scores_max, scores_real_df, clf_df,
                    pickers=PICK_DICT):
    '''
    Return selected image (among a batcf on n images) for each picker.

    scores_df : DataFrame, shape (n, n_discriminators)
    scores_real_df : DataFrame, shape (m, n_discriminators)
    clf_df : Series, shape (n_classifiers x n_calibrators,)
    pickers : dict of str -> callable
    '''
    assert len(scores_df.columns.names) == 1
    assert list(scores_df.columns) == list(scores_real_df.columns)

    init_idx = np.random.choice(len(scores_real_df))

    picked = pd.DataFrame(data=0, index=pickers.keys(), columns=clf_df.index,
                          dtype=int)
    cap_out = pd.DataFrame(data=False,
                           index=pickers.keys(), columns=clf_df.index,
                           dtype=bool)
    alpha = pd.DataFrame(data=np.nan,
                         index=pickers.keys(), columns=clf_df.index,
                         dtype=float)
    for disc_name in sorted(scores_df.columns):
        assert isinstance(disc_name, str)
        s0 = scores_real_df[disc_name].values[init_idx]
        assert np.ndim(s0) == 0
        for calib_name in sorted(clf_df[disc_name].index):
            assert isinstance(calib_name, str)
            calibrator = clf_df[(disc_name, calib_name)]
            s_ = np.concatenate(([s0], scores_df[disc_name].values))
            s_ = calibrator.predict(s_)
            s_max, = calibrator.predict(np.array([scores_max[disc_name]]))
            for picker_name in sorted(pickers.keys()):
                assert isinstance(picker_name, str)
                idx, aa = pickers[picker_name](s_, score_max=s_max)

                if idx == 0:
                    # Try again but init from first fake
                    cap_out.loc[picker_name, (disc_name, calib_name)] = True
                    idx, aa = pickers[picker_name](s_[1:], score_max=s_max)
                else:
                    idx = idx - 1
                assert idx >= 0

                picked.loc[picker_name, (disc_name, calib_name)] = idx
                alpha.loc[picker_name, (disc_name, calib_name)] = aa
    return picked, cap_out, alpha


def enhance_samples_series(g_d_f, scores_real_df, clf_df,
                           pickers=PICK_DICT, n_images=64):
    '''
    Call enhance_samples multiple times to build up a batch of selected images.

    Stores list of used images X separate from the indices of the images
    selected by each method. This is more memory efficient if there are
    duplicate images selected.

    g_d_f : callable returning (X, scores) compliant with `validate`
    calibrator : dict of str -> trained sklearn classifier
        same keys as scores
    n_images : int
    '''
    batch_size = 64   # Batch size to use when calling the pytorch generator G
    chain_batches = 10  # Number of batches to use total for the pickers
    max_est_batches = 156  # Num batches for estimating M in DRS pilot samples

    assert n_images > 0

    _, scores_max = batched_gen_and_disc(g_d_f, max_est_batches, batch_size)
    scores_max = scores_max.max(axis=0)

    print('max scores')
    print(scores_max.to_string())

    X = []
    picked = [None] * n_images
    cap_out = [None] * n_images
    alpha = [None] * n_images
    for nn in xrange(n_images):
        X_, scores_fake_df = \
            batched_gen_and_disc(g_d_f, chain_batches, batch_size)
        picked_, cc, aa = \
            enhance_samples(scores_fake_df, scores_max, scores_real_df, clf_df,
                            pickers=pickers)
        picked_ = picked_.unstack()  # Convert to series

        # Only save the used images for memory, so some index x-from needed
        assert np.ndim(picked_.values) == 1
        used_idx, idx_new = np.unique(picked_.values, return_inverse=True)
        picked_ = pd.Series(data=idx_new, index=picked_.index)

        # A bit of index manipulation in our memory saving scheme
        picked[nn] = len(X) + picked_
        X.extend(list(X_[used_idx]))  # Unravel first index to list
        cap_out[nn] = cc.unstack()
        alpha[nn] = aa.unstack()

    X = np.asarray(X)
    assert X.ndim == 4
    picked = pd.concat(picked, axis=1).T
    assert picked.shape == (n_images, len(picked_))
    cap_out = pd.concat(cap_out, axis=1).T
    assert cap_out.shape == (n_images, len(picked_))
    alpha = pd.concat(alpha, axis=1).T
    assert alpha.shape == (n_images, len(picked_))
    return X, picked, cap_out, alpha


def discriminator_analysis(scores_fake_df, scores_real_df, ref_method,
                           dump_fname=None):
    '''
    scores_fake_df : DataFrame, shape (n, n_discriminators)
    scores_real_df : DataFrame, shape (n, n_discriminators)
    ref_method : (str, str)

    perf_report : str
    calib_report : str
    clf_df : DataFrame, shape (n_calibrators, n_discriminators)
    '''
    # Build combined data set dataframe and train calibrators
    pred_df, y_true = cl.combine_class_df(neg_class_df=scores_fake_df,
                                          pos_class_df=scores_real_df)
    pred_df, y_true, clf_df = cl.calibrate_pred_df(pred_df, y_true)
    # Make methods flat to be compatible with benchmark tools
    pred_df.columns = cl.flat_cols(pred_df.columns)
    ref_method = cl.flat(ref_method)  # Make it flat as well

    # Do calibration analysis
    Z = cl.calibration_diagnostic(pred_df, y_true)
    calib_report = Z.to_string()

    # Dump prediction to csv in case we want it for later analysis
    if dump_fname is not None:
        pred_df_dump = pd.DataFrame(pred_df, copy=True)
        pred_df_dump[LABEL] = y_true
        pred_df_dump.to_csv(dump_fname, header=True, index=False)

    # No compute report on performance of each discriminator:
    # Make it into log-scale cat distn for use with benchmark tools
    pred_df = cl.binary_pred_to_one_hot(pred_df, epsilon=1e-12)
    perf_df, _ = btc.summary_table(pred_df, y_true,
                                   btc.STD_CLASS_LOSS, btc.STD_BINARY_CURVES,
                                   ref_method=ref_method)

    crap_lim = const_dict(1)

    try:
        perf_report = sp.just_format_it(perf_df, shift_mod=3,
                                        crap_limit_min=crap_lim,
                                        crap_limit_max=crap_lim,
                                        EB_limit=crap_lim,
                                        non_finite_fmt={'nan': '--'})
    except Exception as e:
        print(str(e))
        perf_report = perf_df.to_string()
    return perf_report, calib_report, clf_df


def image_dump(X, name, dir_):
    '''This is to isolate torch stuff out.

    X : ndarray, shape (n, nc, image_size, image_size)
    name : str
    '''
    batch_size, nc, image_size, _ = X.shape
    assert X.shape == (batch_size, nc, image_size, image_size)

    fname = os.path.join(dir_, '%s.png' % name)
    vutils.save_image(torch.tensor(X), fname, normalize=True)


def get_inception_score_precomp(all_samples):
    '''
    all_samples : ndarray, shape (n, nc, image_size, image_size)
    '''
    all_samples = np.asarray(all_samples)
    batch_size, nc, image_size, _ = all_samples.shape
    assert all_samples.shape == (batch_size, nc, image_size, image_size)
    assert np.all(-1.0 <= all_samples)
    assert np.all(all_samples <= 1.0)

    if DBG:
        preds = logistic(np.array([x.ravel()[:1000] for x in all_samples]))
    else:
        preds = inception_score_precomp(list(all_samples), resize=True)
    return preds


def get_inception_score_quick(X, picked, n_split=10):
    preds = get_inception_score_precomp(X)

    incep = {}
    for method in picked:
        preds_ = preds[picked[method].values, :]
        incep[(INCEP, method)] = inception_score_fast(preds_, splits=n_split)
    incep = pd.DataFrame(incep)
    incep.columns.names = (METRIC, METHOD)
    return incep


print('start')

# Reference in stat test: base discriminator, raw for no calibration
ref_method = (BASE_D, 'raw')
incep_ref = BASE_D + '_iso_base'
n_split = 80  # Number of splits in inception score
n_score = 64 * n_split  # Total images to generate for benchmark
banner_fmt = '-' * 10 + ' %d ' + '-' * 10

# Get the options, follows pattern of pytorch demo_dcgan.py
opt = get_opts()
batch_size = opt.batchSize
outf = os.path.abspath(os.path.expanduser(opt.outf))
outf = mkdtemp(dir=outf)

print('using dump folder:')
print(outf)

# Reccomended if desire reprod over speed:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set all random seeds: avoid correlated streams ==> must use diff seeds.
master_stream = np.random.RandomState(opt.manualSeed)
py_seed(master_stream.randint(SEED_MAX))
np.random.seed(master_stream.randint(SEED_MAX))
torch.manual_seed(master_stream.randint(SEED_MAX))
if torch.cuda.is_available():
    torch.cuda.manual_seed(master_stream.randint(SEED_MAX))

# Setup python generator that yields function to use the GAN each epoch
device = torch.device('cuda:0' if opt.cuda else 'cpu')
if DBG:  # Just for quick check of code
    print('Warning! Bogus data in debug mode!')
    T = trainer_dummy()
else:
    data_loader = get_data_loader(opt.dataset, opt.dataroot, opt.workers,
                                  opt.imageSize, batch_size)
    T = gan_trainer(device=device, data_loader=data_loader,
                    batch_size=batch_size, nz=opt.nz, ngf=opt.ngf,
                    ndf=opt.ndf, lr=opt.lr, beta1=opt.beta1, ngpu=opt.ngpu,
                    netG_file=opt.netG, netD_file=opt.netD, outf=outf,
                    calib_frac=0.1)

if SKIP_INIT_EVAL:
    # If we don't want to benchmark random initial parameters
    _ = next(T)

agg_perf = {}  # Use dict so that index gets saved on pd concat
for epoch, (_, g_d_f, scores_real) in enumerate(T):
    print(banner_fmt % epoch)

    print('found shapes for real data scores:')
    scores_real_df = validate_scores(scores_real)
    print(scores_real_df.shape)
    n_real_batches, rem = divmod(len(scores_real_df), batch_size)
    assert rem == 0

    # Get some fake scores from generator for the calibration
    _, scores_fake_df = batched_gen_and_disc(g_d_f, n_real_batches, batch_size)

    print('discriminator analysis')
    # Train the calibrator using the real scores dataframe, and fake scores
    # dataframe. There are returned in clf_df. Also return report tables on
    # accuracy of the discriminator and its calibration statistic.
    score_fname = os.path.join(outf, '%d_scores.csv' % epoch)
    perf_report, calib_report, clf_df = \
        discriminator_analysis(scores_fake_df, scores_real_df, ref_method,
                               dump_fname=score_fname)
    print(calib_report)
    print(perf_report)

    if SAVE_IMAGES:
        print('image dumps...')
        # Some image dumps in case we want to actually look at generated images
        X, picked, _, _ = enhance_samples_series(g_d_f, scores_real_df, clf_df)
        picked.columns = cl.flat_cols(picked.columns, name=METHOD)
        for method in picked:
            image_dump(X[picked[method].values], '%d_%s' % (epoch, method),
                       dir_=outf)

    if INCEP_SCORE:
        # Bigger dump for scoring
        print('incep score @ %d' % epoch)
        # X is just a list of images while picked is a dataframe with the
        # indices that each image picked. This is more memory efficient than
        # having the image for each method, since there might be duplicates
        # across pickers.
        X, picked, cap_out, alpha = \
            enhance_samples_series(g_d_f, scores_real_df, clf_df,
                                   n_images=n_score)
        print('%d unique images' % len(X))

        print('cap count:')
        print(cap_out.sum(axis=0).to_string())
        print('accept probs')
        print(alpha.describe().to_string())

        # Get inception scores results
        picked.columns = cl.flat_cols(picked.columns, name=METHOD)
        cap_out.columns = cl.flat_cols(cap_out.columns, name=METHOD)
        incep = get_inception_score_quick(X, picked, n_split)

        # Display nice summary of perf results
        perf_df = bt.loss_summary_table(incep, ref_method=incep_ref,
                                        pairwise_CI=True)
        try:
            perf_report = sp.just_format_it(perf_df, shift_mod=3,
                                            non_finite_fmt={'nan': '--'})
        except Exception as e:
            print(str(e))
            perf_report = perf_df.to_string()
        print(perf_report)

        # Keep concat the perf_df each epoch and write to csv
        agg_perf[epoch] = perf_df
        agg_perf_df = pd.concat(agg_perf, axis=0)
        agg_perf_df.index.names = [('epoch', NA_LEVEL), (METHOD, NA_LEVEL)]
        agg_perf_df.reset_index(drop=False, inplace=True)
        agg_perf_df.to_csv(os.path.join(outf, 'perf.csv'),
                           header=True, index=False)
