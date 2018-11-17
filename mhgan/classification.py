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
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from benchmark_tools.constants import METHOD
import benchmark_tools.classification as btc
import benchmark_tools.data_splitter as ds

LABEL = 'label'


class Calibrator(object):
    def fit(self, y_pred, y_true):
        raise NotImplementedError

    def predict(self, y_pred):
        raise NotImplementedError

    @staticmethod
    def validate(y_pred, y_true=None):
        y_pred = np.asarray(y_pred)
        assert y_pred.ndim == 1
        assert y_pred.dtype.kind == 'f'
        assert np.all(0 <= y_pred) and np.all(y_pred <= 1)

        if y_true is not None:
            y_true = np.asarray(y_true)
            assert y_true.shape == y_pred.shape
            assert y_true.dtype.kind == 'b'

        return y_pred, y_true


class Identity(Calibrator):
    def fit(self, y_pred, y_true):
        assert y_true is not None
        Calibrator.validate(y_pred, y_true)

    def predict(self, y_pred):
        Calibrator.validate(y_pred)
        # Could make copy to be consistent with other methods, but prob does
        # not matter.
        return y_pred


class Linear(Calibrator):
    def __init__(self):
        self.clf = LogisticRegression()

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        self.clf.fit(y_pred[:, None], y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_calib = self.clf.predict_proba(y_pred[:, None])[:, 1]
        return y_calib


class Isotonic(Calibrator):
    def __init__(self):
        self.clf = IsotonicRegression(y_min=0.0, y_max=1.0,
                                      out_of_bounds='clip')

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        self.clf.fit(y_pred, y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_calib = self.clf.predict(y_pred)
        return y_calib


class Beta1(Calibrator):
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.clf = LogisticRegression()

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        y_pred = logit(np.clip(y_pred, self.epsilon, 1.0 - self.epsilon))
        self.clf.fit(y_pred[:, None], y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_pred = logit(np.clip(y_pred, self.epsilon, 1.0 - self.epsilon))
        y_calib = self.clf.predict_proba(y_pred[:, None])[:, 1]
        return y_calib


class Beta2(Calibrator):
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.clf = LogisticRegression()

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        y_pred = np.clip(y_pred.astype(np.float_),
                         self.epsilon, 1.0 - self.epsilon)
        y_pred = np.stack((np.log(y_pred), np.log(1.0 - y_pred)), axis=1)
        self.clf.fit(y_pred, y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_pred = np.clip(y_pred.astype(np.float_),
                         self.epsilon, 1.0 - self.epsilon)
        y_pred = np.stack((np.log(y_pred), np.log(1.0 - y_pred)), axis=1)
        y_calib = self.clf.predict_proba(y_pred)[:, 1]
        return y_calib


CALIB_DICT = {'raw': Identity, 'iso': Isotonic}


def flat(tup, delim='_'):
    '''Join only invertible if delim not in elements.'''
    assert not any(delim in x for x in tup)
    flat_str = delim.join(tup)
    return flat_str


def flat_cols(cols, delim='_', name=None):
    assert isinstance(cols, pd.MultiIndex)
    cols = pd.Index([flat(tup, delim=delim) for tup in cols.values], name=name)
    return cols


def combine_class_df(neg_class_df, pos_class_df):
    '''
    neg_class_df : DataFrame, shape (n, n_features)
    pos_class_df : DataFrame, shape (n, n_features)
        Must have same keys as `neg_class_df`

    df : DataFrame, shape (2 * n, n_features)
    y_true : ndarray, shape (2 * n,)
    '''
    # Adding a new col won't change anything in original
    neg_class_df = pd.DataFrame(neg_class_df, copy=True)
    pos_class_df = pd.DataFrame(pos_class_df, copy=True)

    assert list(neg_class_df.columns) == list(pos_class_df.columns)
    # Note nec always the case, but for now let's require balance
    assert list(neg_class_df.index) == list(pos_class_df.index)
    assert LABEL not in neg_class_df

    neg_class_df[LABEL] = False
    pos_class_df[LABEL] = True

    df = pd.concat((neg_class_df, pos_class_df), axis=0, ignore_index=True)
    y_true = df.pop(LABEL).values

    return df, y_true


def calibrate_pred_df(pred_df, y_true, calib_frac=0.5, calibrators=CALIB_DICT):
    '''
    df : DataFrame, shape (n, n_classifiers)
    y_true : ndarray, shape (n,)
    calib_frac : float
    calibrators : dict of str -> Calibrator

    pred_calib_df : DataFrame, shape (m, n_classifiers x n_calibrators)
        m = calib_frac * n, but rounded
    y_true_test : ndarray, shape (m,)
    clf_df : Series, shape (n_classifiers x n_calibrators,)
    '''
    assert len(pred_df.columns.names) == 1
    assert not pred_df.isnull().any().any()
    assert len(pred_df) == len(y_true)

    idx = ds.rand_mask(len(pred_df), frac=calib_frac)
    y_true_train, y_true_test = y_true[idx], y_true[~idx]

    cols = pd.MultiIndex.from_product([pred_df.columns, calibrators.keys()])
    pred_calib_df = pd.DataFrame(index=xrange(len(y_true_test)), columns=cols,
                                 dtype=float)
    clf_df = pd.Series(index=cols, dtype=object)
    for method in pred_df:
        y_prob = pred_df[method].values
        y_prob_train, y_prob_test = y_prob[idx], y_prob[~idx]
        for calib_name, calib in calibrators.iteritems():
            clf = calib()
            clf.fit(y_prob_train, y_true_train)
            clf_df[(method, calib_name)] = clf

        for calib_name in calibrators:
            pred_calib_df.loc[:, (method, calib_name)] = \
                clf_df[(method, calib_name)].predict(y_prob_test)

    assert not pred_calib_df.isnull().any().any()
    assert pred_calib_df.shape == (len(y_true_test),
                                   len(pred_df.columns) * len(calibrators))
    return pred_calib_df, y_true_test, clf_df


def binary_pred_to_one_hot(df, epsilon=0.0):
    '''
    df : DataFrame, shape (n, n_discriminators)

    df : DataFrame, shape (n, 2 * n_discriminators)
    '''
    assert len(df.columns.names) == 1
    assert not df.isnull().any().any()

    D = {}
    for k in df:
        assert isinstance(k, str)

        x = df[k].values
        x = np.clip(x, epsilon, 1.0 - epsilon)
        D[k] = pd.DataFrame(np.array([np.log(1.0 - x), np.log(x)]).T)
    df_btc = pd.concat(D, axis=1)
    df_btc.columns.names = [METHOD, btc.LABEL]

    assert len(df_btc.columns.names) == 2
    assert df_btc.shape == (df.shape[0], 2 * df.shape[1])
    return df_btc


def calib_score(y_prob, y_true):
    '''
    y_prob : ndarray, shape (n,)
        floats in [0, 1]
    y_true : ndarray, shape (n,)
        bool
    '''
    assert y_true.dtype.kind == 'b'

    Z = np.sum(y_true - y_prob) / np.sqrt(np.sum(y_prob * (1.0 - y_prob)))
    return Z


def calibration_diagnostic(pred_df, y_true):
    calibration_df = pred_df.apply(calib_score, axis=0, args=(y_true,))
    return calibration_df
