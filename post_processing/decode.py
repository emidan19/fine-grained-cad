import argparse
import os
import pickle

from hmmlearn import _hmmc
from librosa.sequence import _viterbi
from librosa.util.exceptions import ParameterError
import numpy as np
import pandas as pd
from scipy.special import logsumexp


NINE_WAY_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
FIVE_WAY_MAP = {0: 4, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 3, 8: 2}
FOUR_WAY_MAP = {0: 3, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 2, 8: 1}


# Modified from librosa
# https://github.com/librosa/librosa/blob/main/librosa/sequence.py
def viterbi(prob, transition, p_state=None, p_init=None, return_logp=False):
    '''Viterbi decoding from discriminative state predictions.'''

    n_states, n_steps = prob.shape

    if transition.shape != (n_states, n_states):
        raise ParameterError('transition.shape={}, must be '
                             '(n_states, n_states)={}'.format(transition.shape,
                                                              (n_states, n_states)))

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ParameterError('Invalid transition matrix: must be non-negative '
                             'and sum to 1 on each row.')

    if np.any(prob < 0) or not np.allclose(prob.sum(axis=0), 1):
        raise ParameterError('Invalid probability values: each column must '
                             'sum to 1 and be non-negative')

    states = np.zeros(n_steps, dtype=int)
    values = np.zeros((n_steps, n_states), dtype=float)
    ptr = np.zeros((n_steps, n_states), dtype=int)

    # Compute log-likelihoods while avoiding log-underflow
    epsilon = np.finfo(prob.dtype).tiny

    # Compute marginal log probabilities while avoiding underflow
    if p_state is None:
        p_state = np.empty(n_states)
        p_state.fill(1./n_states)
    elif p_state.shape != (n_states,):
        raise ParameterError('Marginal distribution p_state must have shape (n_states,). '
                             'Got p_state.shape={}'.format(p_state.shape))
    elif np.any(p_state < 0) or not np.allclose(p_state.sum(axis=-1), 1):
        raise ParameterError('Invalid marginal state distribution: '
                             'p_state={}'.format(p_state))

    log_trans = np.log(transition + epsilon)
    log_marginal = np.log(p_state + epsilon)

    # By Bayes' rule, P[X | Y] * P[Y] = P[Y | X] * P[X]
    # P[X] is constant for the sake of maximum likelihood inference
    # and P[Y] is given by the marginal distribution p_state.
    #
    # So we have P[X | y] \propto P[Y | x] / P[Y]
    # if X = observation and Y = states, this can be done in log space as
    # log P[X | y] \propto \log P[Y | x] - \log P[Y]
    log_prob = np.log(prob.T + epsilon) - log_marginal

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1./n_states)
    elif np.any(p_init < 0) or not np.allclose(p_init.sum(), 1):
        raise ParameterError('Invalid initial state distribution: '
                             'p_init={}'.format(p_init))

    log_p_init = np.log(p_init + epsilon)

    _viterbi(log_prob, log_trans, log_p_init, states, values, ptr)

    if return_logp:
        return states, values[-1, states[-1]]

    return states


def log_normalize(a, axis=None):
    """
    Normalizes the input array so that ``sum(exp(a)) == 1``.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state GMMHMM in the degenerate case normalizing a
        # single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = logsumexp(a, axis, keepdims=True)
        a -= a_lse


# Modified from hmmlearn
# https://github.com/hmmlearn/hmmlearn/blob/master/lib/hmmlearn/base.py
def forward_backward(prob, transition, p_state=None, p_init=None):
    n_states, _ = prob.shape
    # Compute log-likelihoods while avoiding log-underflow
    epsilon = np.finfo(prob.dtype).tiny

    # Compute marginal log probabilities while avoiding underflow
    if p_state is None:
        p_state = np.empty(n_states)
        p_state.fill(1./n_states)
    elif p_state.shape != (n_states,):
        raise ParameterError('Marginal distribution p_state must have shape (n_states,). '
                             'Got p_state.shape={}'.format(p_state.shape))
    elif np.any(p_state < 0) or not np.allclose(p_state.sum(axis=-1), 1):
        raise ParameterError('Invalid marginal state distribution: '
                             'p_state={}'.format(p_state))

    log_trans = np.log(transition + epsilon)
    log_marginal = np.log(p_state + epsilon)

    # By Bayes' rule, P[X | Y] * P[Y] = P[Y | X] * P[X]
    # P[X] is constant for the sake of maximum likelihood inference
    # and P[Y] is given by the marginal distribution p_state.
    #
    # So we have P[X | y] \propto P[Y | x] / P[Y]
    # if X = observation and Y = states, this can be done in log space as
    # log P[X | y] \propto \log P[Y | x] - \log P[Y]
    log_prob = np.log(prob.T + epsilon) - log_marginal

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1./n_states)
    elif np.any(p_init < 0) or not np.allclose(p_init.sum(), 1):
        raise ParameterError('Invalid initial state distribution: '
                             'p_init={}'.format(p_init))

    log_p_init = np.log(p_init + epsilon)
    n_samples, n_components = log_prob.shape

    # Forward
    fwdlattice = np.zeros((n_samples, n_components))
    _hmmc._forward(n_samples, n_components, log_p_init, log_trans, log_prob, fwdlattice)

    # Backward
    bwdlattice = np.zeros((n_samples, n_components))
    _hmmc._backward(n_samples, n_components, log_p_init, log_trans, log_prob, bwdlattice)

    # Compute posteriors
    log_gamma = fwdlattice + bwdlattice
    log_normalize(log_gamma, axis=1)
    posteriors = np.exp(log_gamma)

    return posteriors


def get_decoder(classes, root, split_csv, start_samples, smooth_factor, decoder):
    # Read data and filter to split
    df = pd.read_csv(split_csv)
    df['split'] = df['split'].str.upper()
    df = df.loc[df['split'] == 'TRAIN']
    df = df.reset_index()
    path = lambda r, f: os.path.join(root, '/'.join(r['session'].split('_')), f)
    df['lab'] = df.apply(lambda r: pickle.load(open(path(r, 'labels.pkl'), 'rb')), axis=1)

    label_map = {
        9: NINE_WAY_MAP,
        5: FIVE_WAY_MAP,
        4: FOUR_WAY_MAP
    }[classes]

    labels = []
    for i, r in df.iterrows():
        labels.append(np.vectorize(label_map.__getitem__)(r['lab']))

    T = np.ones((classes, classes)) * smooth_factor * len(labels)
    M = np.ones(classes)            * smooth_factor * len(labels)
    I = np.ones(classes)            * smooth_factor * start_samples
    for l in labels:
        for idx, (i, j) in enumerate(zip(l[:-1], l[1:])):
            if idx < start_samples:
                I[i] += 1
            T[i][j] += 1
            M[i] += 1
        M[l[-1]] += 1

    for c in range(classes):
        T[c] = T[c] / sum(T[c])
    M = M / sum(M)
    I = I / sum(I)

    float_formatter = "{:0.4f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})
    print('ist', 'stu', 'mlt', 'sil', 'oth')
    print('I =', I)
    print('M =', M)
    print('T =\n', T)

    if decoder == 'viterbi':
        return lambda x: viterbi(x, T, p_state=M, p_init=I)
    elif decoder == 'forward-backward':
        return lambda x: forward_backward(x, T, p_state=M, p_init=I)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file')
    parser.add_argument('--smooth', type=float, default=0)
    parser.add_argument('--start_len', type=int, default=500)
    parser.add_argument('--decoder', choices=['viterbi', 'forward-backward'])
    parser.add_argument('--out_file')
    args = parser.parse_args()
    if args.out_file is None:
        out_file = args.in_file.split('.')
        out_file[1] = 'decoded'
        args.out_file = '.'.join(out_file)
    return args


def main():
    args = get_args()

    with open(args.in_file, 'rb') as f:
        res = pickle.load(f)

    classes = next(iter(res.values()))['probs'].shape[0]
    decoder = get_decoder(classes, '/research/hutchinson/workspace/slymane/ml_teaching/pase/dsets/caaml_norm_kaiser_wavmax_3',
                          '/research/hutchinson/data/2019_ml_teaching/split.csv',
                          start_samples=args.start_len, smooth_factor=args.smooth, decoder=args.decoder)

    for session, data in res.items():
        decoded = decoder(data['probs']).transpose()
        res[session]['probs'] = decoded

    with open(args.out_file, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()
