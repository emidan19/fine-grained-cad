import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


NINE_WAY_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
FIVE_WAY_MAP = {0: 4, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 3, 8: 2}
FOUR_WAY_MAP = {0: 3, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 2, 8: 1}


class CAAMLRawFrameDataset(Dataset):
    def __init__(self, split,  root, classes=9, seq_len=10*100, sample_rate=16_000, mask=None, mask_mode=None,
                 spacing=0, split_csv='/research/hutchinson/data/2019_ml_teaching/split.csv', return_session=False,
                 features='pase'):

        # Read data and filter to split
        df = pd.read_csv(split_csv)
        df['split'] = df['split'].str.upper()
        df = df.loc[df['split'] == split.upper()]
        df = df.reset_index()

        # Construct wav path and labels
        mel_root = '/research/hutchinson/data/2019_ml_teaching/mel_features/normalized/'
        path = lambda r, f: os.path.join(root, '/'.join(r['session'].split('_')), f)
        prep = lambda r: os.path.join(mel_root, r['session'] + '.npy')
        df['wav'] = df.apply(lambda r:                  path(r, 'signal.npy'),         axis=1)
        df['pre'] = df.apply(lambda r:                  prep(r),                       axis=1)
        df['lab'] = df.apply(lambda r: pickle.load(open(path(r, 'labels.pkl'), 'rb')), axis=1)
        df['len'] = df.apply(lambda r: len(r['lab']),                                  axis=1)
        df['start'] = 0
        df['end'] = 0

        label_map = {
            9: NINE_WAY_MAP,
            5: FIVE_WAY_MAP,
            4: FOUR_WAY_MAP
        }[classes]

        # Create a indexing scheme for all possible sequence start positions
        padding = (seq_len-spacing) // 2
        l = 0
        wavs = {}
        pres = {}
        for i, r in df.iterrows():
            df.loc[i, 'start'] = l
            l += r['len'] + 2*padding
            rounding = (spacing - l % spacing) - 1
            l += rounding
            df.loc[i, 'len'] = r['len'] + 2*padding + rounding
            df.loc[i, 'end'] = l

            labs = np.vectorize(label_map.__getitem__)(r['lab'])
            labs = np.concatenate([-np.ones(padding, dtype=np.long), labs, -np.ones(seq_len+rounding+padding,
                                  dtype=np.long)])

            sigs = np.load(r['wav'])
            if 'pase' in features:
                prepend = sigs[:, :160*padding][::-1] if padding > 0 else None
                pstpend = sigs[:, -160*(padding+rounding+seq_len):][::-1]
                sigs = np.concatenate([prepend, sigs, pstpend], axis=1) if prepend is not None else np.concatenate([sigs, pstpend], axis=1)
            else:
                sigs = None

            prec = np.load(r['pre'], allow_pickle=True)
            if 'mels' in features and 'prosody' in features:
                prec = np.concatenate(prec, axis=0)
            elif 'mels' in features:
                prec = prec[0]
            elif 'prosody' in features:
                prec = prec[1]
            else:
                prec = None

            if prec is not None:
                prepend = prec[:, :padding][::-1] if padding > 0 else None
                pstpend = prec[:, -(padding+rounding+seq_len):][::-1]
                prec = np.concatenate([prepend, prec, pstpend], axis=1) if prepend is not None else np.concatenate([prec, pstpend], axis=1)
                prec = prec[:, :len(labs)]

            df.at[i, 'lab'] = labs
            wavs[r['wav']] = sigs
            pres[r['wav']] = prec
            l += 1

        self.data = df
        self.wavs = wavs
        self.pres = pres
        self.n_samples = l
        self.seq_len = seq_len
        self.mask = mask
        self.mask_mode = mask_mode
        self.return_session = return_session

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Retrieve row from dataframe
        r = self.data.loc[(self.data['start'] <= idx) & (self.data['end'] >= idx)]
        wav = self.wavs[r['wav'].item()]
        pre = self.pres[r['wav'].item()]
        lab = r['lab'].item()

        # Get appropriate sequence
        seq_start = idx - r['start'].item()
        seq_end = seq_start + self.seq_len
        lab = lab[seq_start:seq_end] if lab is not None else np.empty(0)
        pre = pre[:, seq_start:seq_end] if pre is not None else np.empty(0)
        wav = wav[:, seq_start*160:seq_end*160] if wav is not None else np.empty(0)

        if self.mask is not None and self.mask_mode is not None:
            time_mask = torch.FloatTensor(wav.shape).uniform_() < self.mask
            if self.mask_mode == 'silence':
                wav[time_mask] = 0
            elif self.mask_mode == 'noise':
                wav[time_mask] = torch.tanh(torch.FloatTensor(wav.shape).normal_()[time_mask])
            elif self.mask_mode == 'sample':
                wav[time_mask] = torch.LongTensor(wav.shape).random_(0, wav.size(1))[time_mask]

        return (wav, pre, lab) if not self.return_session else (wav, pre, lab, r['session'].item())


class SpacedSequentialSampler(Sampler):
    def __init__(self, data_source, spacing=1):
        self.data_source = data_source
        self.spacing = spacing

    def __iter__(self):
        n = len(self.data_source)
        indicies = [i for i in range(n)][::self.spacing]
        return iter(indicies)

    def __len__(self):
        return len(self.data_source) // self.spacing


class SpacedRandomSampler(Sampler):
    def __init__(self, data_source, spacing=1):
        self.data_source = data_source
        self.spacing = spacing
        self.offset = 0
        self.n_samples = len(self.data_source) // self.spacing

    def __iter__(self):
        n = len(self.data_source)
        indicies = [i for i in range(n)][self.offset::self.spacing]
        self.n_samples = len(indicies)
        self.offset = random.randrange(self.spacing)
        random.shuffle(indicies)
        return iter(indicies)

    def __len__(self):
        return self.n_samples


class SessionBatchSampler(Sampler):
    def __init__(self, dataset, sampler, batch_size, drop_last):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        _, _, _, cur_session = self.dataset[0]
        for idx in self.sampler:
            _, _, _, session = self.dataset[idx]
            if cur_session != session and len(batch) != 0:
                yield batch
                batch = []

            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            cur_session = session

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
