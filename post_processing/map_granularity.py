import argparse
import pickle

import numpy as np


FIVE_WAY_MAP = {0: 4, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 3, 8: 2}
FOUR_WAY_MAP = {0: 3, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 2, 8: 1}
NINE_WAY_MAP = {
    0: 1,  # o  -> a
    1: 2,  # a  -> l
    2: 2,  # l  -> iq
    3: 3,  # iq -> ia
    4: 4,  # ia -> sq
    5: 5,  # sq -> sa
    6: 6,  # sa -> g
    7: 7,  # s  -> s
    8: 8   # g  -> o
}

A = ['o', 'a', 'l',  'iq', 'ia', 'sq', 'sa', 's', 'g']
B = ['a', 'l', 'iq', 'ia', 'sq', 'sa', 'g',  's', 'o']
NINE_WAY_MAP = {i: B.index(v) for i, v in enumerate(A)}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file', required=True, help='Input predictions file to map')
    parser.add_argument('-w', '--way', type=int, required=True, choices=[4, 5, 9], help='Way to map down to')
    parser.add_argument('-o', '--out_file', help='File to output to', default='predictions.mapped.pkl')
    return parser.parse_args()


def main():
    args = get_args()
    sessions = np.load(args.in_file, allow_pickle=True)
    label_map = {
        4: FOUR_WAY_MAP,
        5: FIVE_WAY_MAP,
        9: NINE_WAY_MAP
    }[args.way]

    mapped_sessions = {}
    for name, data in sessions.items():
        # Infer the way from the map and create empty storage array
        way = len({v for v in label_map.values()})
        mapped_probs = np.zeros((way, data['probs'].shape[1]))

        # Map probabilities from 9-way into way. Summing together mass mapped to same label
        for idx in range(9):
            mapped_probs[label_map[idx]] += data['probs'][idx]

        # Make sure the probabilities in each frame still sum to one
        assert np.allclose(np.sum(mapped_probs, axis=0), np.ones(mapped_probs.shape[1]))

        # Map the labels. Vectorize is just a fast way to do this
        mapped_labs = np.vectorize(label_map.__getitem__)(data['labs'])

        # Create and save the mapped version
        mapped_sessions[name] = {'labs': mapped_labs, 'probs': mapped_probs}

    with open(args.out_file, 'wb') as f:
        pickle.dump(mapped_sessions, f)


if __name__ == '__main__':
    main()
