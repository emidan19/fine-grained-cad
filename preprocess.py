import glob
import os
import pickle
import shutil

import librosa
import numpy as np
import tqdm


DEF_CAAML_ROOT = '/research/hutchinson/data/2019_ml_teaching'
DEF_NORM_ROOT = 'caaml_norm_kaiser_wavmax_3'
LABEL_DICT = {'o': 0, 'a': 1, 'l': 2, 'iq': 3, 'ia': 4, 'sq': 5, 'sa': 6, 's': 7, 'g': 8}


def unroll_labels(path, frame_len=0.15, offset=0.01):
    with open(path, 'r') as a:
        lines = a.readlines()

    unrolled_labels = []
    cur_time = 0.0
    for i in range(1, len(lines) + 1):
        if i == len(lines):
            end = float(lines[i-1].split()[1])
            while cur_time < end:
                if len(lines[i-1].split()) != 3:
                    print(f'{path} is improperly formatted')
                    break
                label_embedding = LABEL_DICT[lines[i-1].split()[2]]
                unrolled_labels.append(label_embedding)
                cur_time += offset
            continue

        prev_end = float(lines[i-1].split()[1])
        start = float(lines[i].split()[0])

        # Copy current label until cur_time > end
        while cur_time + offset < prev_end:
            if len(lines[i-1].split()) != 3:
                print(f'{path} is improperly formatted')
                break
            label_embedding = LABEL_DICT[lines[i-1].split()[2]]
            unrolled_labels.append(label_embedding)
            cur_time += offset

        # Deal with overlap
        mid = cur_time + (frame_len / 2)
        if mid < prev_end:
            if len(lines[i-1].split()) != 3:
                print(f'{path} is improperly formatted')
                break
            label_embedding = LABEL_DICT[lines[i-1].split()[2]]
            unrolled_labels.append(label_embedding)
        elif mid > start:
            if len(lines[i-1].split()) != 3:
                print(f'{path} is improperly formatted')
                break
            label_embedding = LABEL_DICT[lines[i-1].split()[2]]
            unrolled_labels.append(label_embedding)
        cur_time += offset
    return unrolled_labels


def load_norm(path, new_sr=16_000):
    signal, sr = librosa.load(path, sr=new_sr)
    signal = signal.reshape(-1, 1)
    signal = signal.T
    return signal, new_sr


def extract(src, dst, glob_string='???/*/*/*.wav', overwrite=False):
    if os.path.exists(dst):
        if overwrite:
            shutil.rmtree(dst)
        else:
            print(f'{dst} already exists, exiting...')
            exit(-1)

    for wpath in tqdm.tqdm(glob.glob(f'{src}/{glob_string}')):
        # Build directory
        rel_path = os.path.dirname(wpath).split('/', maxsplit=src.count('/')+1)[-1]
        norm_path = os.path.join(dst, rel_path)
        os.makedirs(norm_path)

        # Get normalized audio
        signal, _ = load_norm(wpath)
        n_frames = signal.shape[1] // 160

        # Get unrolled labels
        apath = os.path.join(os.path.dirname(wpath), 'annotations0.txt')
        labels = unroll_labels(apath)

        # Ensure valid bounds on data
        n_frames = min(n_frames, len(labels))
        signal = signal[:, :160*n_frames]
        labels = labels[:n_frames]

        # Save data
        shutil.copy(apath, norm_path)
        np.save(os.path.join(norm_path, 'signal.npy'), signal)
        with open(os.path.join(norm_path, 'labels.pkl'), 'wb') as f:
            pickle.dump(labels, f)

        assert(n_frames == len(labels))
        assert(n_frames == signal.shape[1] / 160)


if __name__ == '__main__':
    extract(DEF_CAAML_ROOT, DEF_NORM_ROOT, '???/*/*/*.wav', overwrite=True)
