import argparse
import pickle

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, nargs='+')
    parser.add_argument('--method', type=str, choices=['avg', 'max'])
    parser.add_argument('--out')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Load all provided predictions
    res = []
    for p in args.predictions:
        with open(p, 'rb') as f:
            res.append(pickle.load(f))

    # Ensure that all predictions are aligned
    for x, y in zip(res[:-1], res[1:]):
        assert x.keys() == y.keys(), 'Failed to match all keys in provided predictions'
        for session in x:
            assert np.all(x[session]['labs'] == y[session]['labs']), 'One or more predictions are not properly aligned'

    # Calculate ensembled predictions
    sessions = list(res[0].keys())
    predictions = {}
    for session in sessions:
        p = [r[session]['probs'] for r in res]

        if args.method == 'avg':
            p = np.mean(np.stack(p), axis=0)
        elif args.method == 'max':
            p = np.max(np.stack(p), axis=0)
            p = p / np.sum(p, axis=0)
        elif args.method == 'frq':
            raise NotImplementedError('Frequency based ensembling not yet implemented')
        else:
            raise TypeError('Unrecognized ensemble type: ', args.method)

        assert np.allclose(np.sum(p, axis=0), np.ones(p.shape[1])), "Ensembled probabilities don't sum to one"
        predictions[session] = {
            'probs': p,
            'labs': res[0][session]['labs']
        }

    with open(args.out, 'wb') as f:
        pickle.dump(predictions, f)


if __name__ == '__main__':
    main()
