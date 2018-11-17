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
import numpy as np
from scipy.special import logit
from scipy.special import expit as logistic


def cumargmax(x):
    '''why is this not in numpy? This may not work with nans.

    This uses last occurence not first occurence like argmax.
    '''
    x = np.asarray(x)

    assert x.ndim == 1

    mm = np.maximum.accumulate(x)
    idx = np.arange(len(x))
    idx[x < mm] = 0
    idx = np.maximum.accumulate(idx)
    return idx

# ============================================================================
# General utils for MCMC
# ============================================================================


def binary_posterior(P0, P1):
    '''Get posterior on P(case 1|x) given likelihoods for case 0 and case 1, P0
    and P1, resp.
    '''
    posterior_1 = np.true_divide(P1, P1 + P0)
    return posterior_1


def disc_2_odds_ratio(P_disc):
    odds_ratio = 1.0 / ((1.0 / P_disc) - 1.0)
    return odds_ratio


def odds_ratio_2_disc(odds_ratio):
    P_disc = 1.0 / ((1.0 / odds_ratio) + 1.0)
    return P_disc


def accept_prob_MH(p_last, p_new, q_last_from_new, q_new_from_last):
    alpha = np.true_divide(p_new * q_last_from_new, p_last * q_new_from_last)
    alpha = np.minimum(1.0, alpha)
    return alpha


def accept_prob_MH_disc(P_disc_last, P_disc_new):
    # These asserts will also catch accidently using log or logistic probs
    assert np.all(0.0 <= P_disc_last) and np.all(P_disc_last <= 1.0)
    assert np.all(0.0 <= P_disc_new) and np.all(P_disc_new <= 1.0)

    alpha = ((1.0 / P_disc_last) - 1.0) / ((1.0 / P_disc_new) - 1.0)
    # Use fmin so get alpha=1 on nan which results from 0 or 1 P_disc values
    # unclear which is the best value for alpha at the discontinuities.
    alpha = np.fmin(1.0, alpha)
    return alpha


def test_accept_prob_MH_disc():
    N = 10

    p_real_last = np.abs(np.random.randn(N))
    p_real_new = np.abs(np.random.randn(N))
    p_gan_last = np.abs(np.random.randn(N))
    p_gan_new = np.abs(np.random.randn(N))

    alpha0 = accept_prob_MH(p_real_last, p_real_new, p_gan_last, p_gan_new)

    P_disc_last = binary_posterior(p_gan_last, p_real_last)
    P_disc_new = binary_posterior(p_gan_new, p_real_new)
    alpha1 = accept_prob_MH_disc(P_disc_last, P_disc_new)

    assert(np.allclose(alpha0, alpha1))

# ============================================================================
# The GAN pickers
# ============================================================================


def rejection_sample(d_score, epsilon=1e-6, shift_percent=95.0, score_max=None,
                     random=np.random):
    '''Rejection scheme from:
    https://arxiv.org/pdf/1810.06758.pdf
    '''
    assert(np.ndim(d_score) == 1 and len(d_score) > 0)
    assert(0 <= np.min(d_score) and np.max(d_score) <= 1)
    assert(np.ndim(score_max) == 0)

    # Chop off first since we assume that is real point and reject does not
    # start with real point.
    d_score = d_score[1:]

    # Make sure logit finite
    d_score = np.clip(d_score.astype(np.float), 1e-14, 1 - 1e-14)
    max_burnin_d_score = np.clip(score_max.astype(np.float),
                                 1e-14, 1 - 1e-14)

    log_M = logit(max_burnin_d_score)

    D_tilde = logit(d_score)
    # Bump up M if found something bigger
    D_tilde_M = np.maximum(log_M, np.maximum.accumulate(D_tilde))

    D_delta = D_tilde - D_tilde_M
    F = D_delta - np.log(1 - np.exp(D_delta - epsilon))

    if shift_percent is not None:
        gamma = np.percentile(F, shift_percent)
        F = F - gamma

    P = logistic(F)
    accept = random.rand(len(d_score)) <= P

    if np.any(accept):
        idx = np.argmax(accept)  # Stop at first true, default to 0
    else:
        idx = np.argmax(d_score)  # Revert to cherry if no accept

    # Now shift idx because we took away the real init point
    return idx + 1, P[idx]


def _mh_sample(d_score, init_picked=0, start=1, random=np.random):
    '''Same as `mh_sample` but more obviously correct.
    '''
    assert(np.ndim(d_score) == 1 and len(d_score) > 0)
    assert(0 <= np.min(d_score) and np.max(d_score) <= 1)
    assert(init_picked < start)

    d_last = np.float_(d_score[init_picked])
    picked_round = init_picked
    for ii, d_new in enumerate(d_score[start:], start):
        d_new = np.float_(d_new)

        # Note: we might want to move to log or logit scale for disc probs if
        # this starts to create numerics issues.
        alpha = accept_prob_MH_disc(d_last, d_new)
        assert(0 <= alpha and alpha <= 1)
        if random.rand() <= alpha:
            d_last = d_new
            picked_round = ii
    return picked_round


def mh_sample(d_score, score_max=None, random=np.random):
    assert(np.ndim(d_score) == 1 and len(d_score) > 0)
    assert(0 <= np.min(d_score) and np.max(d_score) <= 1)

    OR = disc_2_odds_ratio(d_score)
    OR_U = OR / random.rand(len(d_score))

    picked_round = 0
    alpha = 1.0
    for ii in xrange(1, len(d_score)):
        if OR[picked_round] <= OR_U[ii]:
            alpha = accept_prob_MH_disc(d_score[picked_round], d_score[ii])
            picked_round = ii
    return picked_round, alpha


def cumm_mh_sample_distn(d_score, w):
    assert(np.ndim(d_score) == 1 and len(d_score) > 0)
    assert(0 <= np.min(d_score) and np.max(d_score) <= 1)
    assert(np.shape(d_score) == np.shape(w))

    rounds = len(d_score)

    p_idx = np.zeros(rounds)
    p_idx[0] = 1.0
    val = np.zeros(rounds)
    val[0] = w[0]
    for ii in xrange(1, rounds):
        P_disc_new = np.float_(d_score[ii])

        # Note this is bit of wasted computation here since p_idx[ii:] == 0
        alpha = accept_prob_MH_disc(d_score, P_disc_new)
        p_idx = p_idx * (1.0 - alpha)
        assert(np.all(p_idx[ii:] == 0.0))

        p_idx[ii] = 1.0 - np.sum(p_idx)
        assert(np.all(0 <= p_idx))
        assert(np.isclose(np.sum(p_idx), 1.0))

        val[ii] = np.dot(p_idx, w)
    return val


if __name__ == '__main__':
    from time import time

    np.random.seed(4534)

    t0 = 0.0
    t1 = 0.0
    for rr in xrange(1000):
        test_accept_prob_MH_disc()  # Run this test too

        d_score = np.random.rand(640)
        seed = np.random.randint(100000)

        r = np.random.RandomState(seed)
        _ = r.rand()
        t = time()
        idx = _mh_sample(d_score, random=r)
        t0 += (time() - t)

        r = np.random.RandomState(seed)
        t = time()
        idx2, _ = mh_sample(d_score, random=r)
        t1 += (time() - t)

        print idx, idx2
        assert idx == idx2
