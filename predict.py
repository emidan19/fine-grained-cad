#!/usr/bin/env python3

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb

from data import CAAMLRawFrameDataset, SpacedSequentialSampler, SessionBatchSampler
from models import PASEEncodedModel, LSTMHead, CNNHead, GRUHead
from utils import get_channel_progression


PACE_EMB_DIM = 256
MEL_DIM = 40
PROSODY_DIM = 3


def get_args():
    parser = argparse.ArgumentParser()

    ###############################
    # GENERAL PARAMETERS
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help="Device to run model on")
    parser.add_argument('--out', default='./predictions.raw.pkl')
    parser.add_argument('--split_id')
    parser.add_argument('--run')
    parser.add_argument('--ckpt_type', default='loss', choices=['err', 'loss', 'f1', 'mAP'])
    parser.add_argument('--mb', type=int, default=8, help='mini-batch size')

    ###############################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('--datapath', type=str, default='/research/hutchinson/workspace/slymane/pase/caaml_norm/')
    data_group.add_argument('--split', type=str, default='/research/hutchinson/data/2019_ml_teaching/split.csv')
    data_group.add_argument('--warmup', type=int, default=0)
    data_group.add_argument('--nworkers', type=int, default=0)

    args = parser.parse_args()

    args.warmup *= 2
    return args


def main():
    # Setup
    args = get_args()

    ckpt_dir = os.path.join('checkpoints', args.run.split('/')[-1])
    ckpt = f'model_best_{args.ckpt_type}.pkl'
    run = wandb.Api().run(args.run)
    config = run.config
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    ckpt_path = os.path.join(ckpt_dir, ckpt)
    if not os.path.exists(ckpt_path):
        wandb.restore(ckpt, run_path=args.run, root=ckpt_dir)

    input_dimension = 0
    precomputed_dim = 0
    if 'pase' in config['features']:
        input_dimension += PACE_EMB_DIM
    if 'mels' in config['features']:
        input_dimension += MEL_DIM
        precomputed_dim += MEL_DIM
    if 'prosody' in config['features']:
        input_dimension += PROSODY_DIM
        precomputed_dim += PROSODY_DIM

    # Model
    if config['head'] == 'mlp':
        cls_head = CNNHead(input_dimension, config['classes'], 1,
                           [config['hidden_size']] * (config['hidden_layers'] - 1),
                           config['smooth'], config['context_size'], 1, config['norm'], config['drop_hid'])
    elif config['head'] == 'dtcnn':
        cls_head = CNNHead(input_dimension, config['classes'], config['dilation_factor'],
                           get_channel_progression(config['hidden_size'], config['hidden_layers'], update_rules=[1, 2]),
                           config['smooth'], config['context_size'], config['context_size'], config['norm'], config['drop_hid'])
    elif config['head'] == 'lstm':
        cls_head = LSTMHead(input_dimension, config['classes'], config['hidden_size'], config['hidden_layers'],
                            config['smooth'], False, config['drop_hid'])
    elif config['head'] == 'bilstm':
        cls_head = LSTMHead(input_dimension, config['classes'], config['hidden_size']//2, config['hidden_layers'],
                            config['smooth'], True, config['drop_hid'])
    elif config['head'] == 'gru':
        cls_head = GRUHead(input_dimension, config['classes'], config['hidden_size'], config['hidden_layers'],
                           config['smooth'], False, config['drop_hid'])
    elif config['head'] == 'bigru':
        cls_head = GRUHead(input_dimension, config['classes'], config['hidden_size']//2, config['hidden_layers'],
                           config['smooth'], True, config['drop_hid'])

    model = PASEEncodedModel(cls_head, config['cfg'], config['ckpt'], drop_inp=config['drop_inp'],
                             drop_emb=config['drop_emb'], freeze_bn=config['freeze_bn'], tune=False).to(args.device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    if args.device == 'cpu':
        model.encoder.rnn.layers[0].use_cuda = False

    # Data
    with torch.no_grad():
        x = torch.rand(1, 1, 160*config['seqlen']).to(args.device) if 'pase' in config['features'] else None
        p = torch.rand(1, precomputed_dim, config['seqlen']).to(args.device)
        spacing = model(x, precomputed=p).detach().cpu().size(2)
    data = CAAMLRawFrameDataset(args.split_id, args.datapath, classes=config['classes'],
                                seq_len=config['seqlen']+args.warmup, split_csv=args.split,
                                return_session=True, spacing=spacing, features=config['features'])
    sampler = SpacedSequentialSampler(data, spacing=spacing)
    batch_sampler = SessionBatchSampler(data, sampler, args.mb, False)
    loader = torch.utils.data.DataLoader(data, batch_sampler=batch_sampler, pin_memory=True, num_workers=args.nworkers)

    # Evaluate
    probs_lst, labs_lst, sessions = [], [], []
    logits_list, labs_list = [], []
    _, _, _, cur_session = data[0]
    with torch.no_grad():
        for idx, (sigs, prec, labs, session) in enumerate(tqdm.tqdm(loader)):
            sigs = sigs.to(args.device).float() if sigs.nelement() != 0 else None
            prec = prec.to(args.device).float() if prec.nelement() != 0 else None
            labs = labs.to(args.device)

            # Forward
            logits = model(sigs, precomputed=prec)

            # Get prediction window for logits
            offset = (logits.size(2) - spacing) // 2
            logits = logits.narrow(2, offset, spacing)

            # Get prediction window for labels
            offset = (labs.size(1) - spacing) // 2
            labs = labs.narrow(1, offset, spacing)

            # Reshape into one long sequence
            logits = logits.transpose(0, 1).reshape(config['classes'], -1)  # N,C,S -> C,N,S -> C,N*S
            labs = labs.reshape(-1)                                         # N,S   -> N*S

            ignore = (labs == -1).squeeze()
            labs = labs[~ignore]
            logits = logits[:, ~ignore]

            if labs.size(0) == 0:
                continue

            if session[0] == cur_session:
                logits_list.append(logits.squeeze().detach().cpu())
                labs_list.append(labs.squeeze().detach().cpu())
            else:
                logits_list = torch.cat(logits_list, dim=1)
                labs_list = torch.cat(labs_list, dim=0)
                sessions.append(cur_session)

                # Calculate metrics
                probs = F.softmax(logits_list, dim=0).cpu().numpy()
                tqdm.tqdm.write(f'Finished {cur_session}')

                # Store metrics
                probs_lst.append(probs)
                labs_lst.append(labs_list.numpy())

                logits_list = [logits.squeeze().detach().cpu()]
                labs_list = [labs.squeeze().detach().cpu()]
                cur_session = session[0]

        logits_list = torch.cat(logits_list, dim=1)
        labs_list = torch.cat(labs_list, dim=0)
        sessions.append(cur_session)

        # Calculate metrics
        probs = F.softmax(logits_list, dim=0).cpu().numpy()
        tqdm.tqdm.write(f'Finished {cur_session}')

        # Store metrics
        probs_lst.append(probs)
        labs_lst.append(labs_list.numpy())

        logits_list = [logits.squeeze().detach().cpu()]
        labs_list = [labs.squeeze().detach().cpu()]
        cur_session = session[0]

    # Collect all predictions
    probs = np.concatenate(probs_lst, axis=1).transpose(1, 0)

    # Save the outputted sequences
    res = {session: {
        'probs': probs_lst[i],  # ([CxS] ndarray) Raw Probabilites
        'labs': labs_lst[i]    # ([S]   ndarray) Ground Truth Labels
    } for i, session in enumerate(sessions)}

    with open(args.out, 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
