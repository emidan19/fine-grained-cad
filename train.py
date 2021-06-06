#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
import tqdm
import wandb

from data import CAAMLRawFrameDataset, SpacedSequentialSampler, SpacedRandomSampler
from metrics import CAAMLMetrics
from models import PASEEncodedModel, LSTMHead, CNNHead, GRUHead
from utils import get_channel_progression, get_accumulation_iters
from losses import cb_loss


PACE_EMB_DIM = 256
MEL_DIM = 40
PROSODY_DIM = 3


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    ###############################
    # GENERAL PARAMETERS
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help="Device to run model on")
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--track', type=str2bool, nargs='?', const=True, default=False,
                        help='Track experiments via wandb')
    parser.add_argument('--timeout', type=float, default=1, help="Days until training times out")

    ###############################
    # OPTIMIZATION PARAMETERS
    train_group = parser.add_argument_group('TRAINING PARAMETERS')
    train_group.add_argument('--epochs', type=int, default=10, help='number of total epochs to run')
    train_group.add_argument('--mb', type=int, default=16, help='mini-batch size')
    train_group.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    train_group.add_argument('--wd', type=float, default=1e-2,
                             help='weight decay (default: 1e-2) weight_decay for L2 regularization.')
    train_group.add_argument('--amsgrad', type=str2bool, nargs='?', const=True, default=False,
                             help='Use the AMSGrad varient of AdamW')
    train_group.add_argument('--patience', type=int, default=9)
    train_group.add_argument('--lrs_patience', type=int, default=3)
    train_group.add_argument('--lrs_factor', type=float, default=0.5)
    train_group.add_argument('--clip_norm', type=float, default=1.0)
    train_group.add_argument('--beta', type=float, choices=[0, 0.9, 0.99, 0.999, 0.9999], default=0.0)
    train_group.add_argument('--gamma', type=float, default=0.0)

    ###############################
    # AUGMENTATION PARAMETERS
    train_group.add_argument('--mask', type=float, default=0.0)
    train_group.add_argument('--mask_mode', type=str, choices=['silence', 'noise', 'sample'])

    ###############################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')

    # Pace only
    model_group.add_argument('--tune', type=str2bool, nargs='?', const=True, default=False)
    model_group.add_argument('--cfg', type=str, default='cfg/PASE+.cfg')
    model_group.add_argument('--ckpt', type=str, default='checkpoints/FE_e199.ckpt')

    # All classification heads
    model_group.add_argument('--head', type=str, required=True,
                             choices=['mlp', 'dtcnn', 'lstm', 'bilstm', 'gru', 'bigru'])
    model_group.add_argument('--freeze_bn', type=str2bool, nargs='?', const=True, default=False)
    model_group.add_argument('--hidden_size', type=int, default=256)
    model_group.add_argument('--hidden_layers', type=int, default=1)
    model_group.add_argument('--drop_inp', type=float, default=0.0)
    model_group.add_argument('--drop_emb', type=float, default=0.0)
    model_group.add_argument('--drop_hid', type=float, default=0.0)
    model_group.add_argument('--smooth', type=int, default=1)

    # mlp and dtcnn only (otherwise None)
    model_group.add_argument('--context_size', type=int, default=1)
    model_group.add_argument('--norm', type=str, choices=['bnorm', 'lnorm', 'inorm', 'affinorm', 'none'], default='none')

    # dtcnn only (otherwise None)
    model_group.add_argument('--dilation_factor', type=float, default=1.0)

    ###############################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('--datapath', type=str, default='/research/hutchinson/workspace/slymane/ml_teaching/pase/dsets/caaml_norm_kaiser_wavmax_3/')
    data_group.add_argument('--split', type=str, default='/research/hutchinson/data/2019_ml_teaching/split.csv')
    data_group.add_argument('--classes', type=int, default=9)
    data_group.add_argument('--seqlen', type=int, default=600)  # max seqlen is about 10,000 (100seconds) on gtx 1080ti
    data_group.add_argument('--nworkers', type=int, default=0)
    data_group.add_argument('--features', type=str, nargs='+', default=['pase'], choices=['pase', 'mels', 'prosody'])

    args = parser.parse_args()

    if args.head in ['lstm', 'bilstm', 'gru', 'bigru', 'mlp']:
        args.dilation_factor = None

    if args.head in ['lstm', 'bilstm', 'gru', 'bigru']:
        args.context_size = None
        args.norm = 'none'

    if args.norm == 'none':
        args.norm = None

    args.timeout = int(args.timeout * 60 * 60 * 24)

    return args


def train(model, criterion, data, n_acc_iters, optimizer, clip_norm, device):
    err_lst, loss_lst = [], []
    for idx, (sigs, prec, labs) in enumerate(tqdm.tqdm(data, leave=False, position=1)):
        sigs = sigs.to(device).float() if sigs.nelement() != 0 else None
        prec = prec.to(device).float() if prec.nelement() != 0 else None
        labs = labs.to(device)

        # Predict
        logits = model(sigs, precomputed=prec)
        logits, labs = filter_predictions(logits, labs)

        if labs.size(0) != 0:
            # Calculate loss
            loss = criterion(logits, labs.unsqueeze(1))
            loss.backward()
            grad = True

            # Calculate metrics
            _, err = get_metrics(logits, labs)

            # Store metrics
            loss_lst.append(loss.detach().cpu().item())
            err_lst.append(err.detach().cpu().item())

        if (idx+1) % n_acc_iters == 0:
            # Optimize
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            grad = False

        del logits

    if grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    trn_err_avg = sum(err_lst) / len(err_lst)
    trn_loss_avg = sum(loss_lst) / len(loss_lst)
    return trn_err_avg, trn_loss_avg


def evaluate(model, criterion, data, n_acc_iters, device):
    err_lst, loss_lst, probs_lst, labs_lst = [], [], [], []
    with torch.no_grad():
        for idx, (sigs, prec, labs) in enumerate(tqdm.tqdm(data, leave=False, position=1)):
            sigs = sigs.to(device).float() if sigs.nelement() != 0 else None
            prec = prec.to(device).float() if prec.nelement() != 0 else None
            labs = labs.to(device)

            # Predict then evalaute loss
            logits = model(sigs, precomputed=prec)
            logits, labs, = filter_predictions(logits, labs)

            if labs.size(0) != 0:
                # Calculate loss
                loss = criterion(logits, labs.unsqueeze(1))

                # Calculate metrics
                probs, err = get_metrics(logits, labs)

                # Store metrics
                loss_lst.append(loss.detach().item())
                err_lst.append(err.detach().item())
                probs_lst.append(probs.detach().cpu().numpy())
                labs_lst.append(labs.detach().cpu().numpy())

            del logits

    dev_err_avg = sum(err_lst) / len(err_lst)
    dev_loss_avg = sum(loss_lst) / len(loss_lst)
    labels = np.concatenate(labs_lst)
    probs = np.concatenate(probs_lst)

    return dev_err_avg, dev_loss_avg, labels, probs


def filter_predictions(logits, labels):
    offset = (labels.size(1) - logits.size(2)) // 2
    labels = labels.narrow(1, offset, logits.size(2))

    logits = logits.transpose(1, 2)              # N,C,S -> N,S,C
    logits = logits.reshape(-1, logits.size(2))  # N,S,C -> N*S,C
    labels = labels.reshape(-1)                  # N,S   -> N*S

    ignore = (labels == -1).squeeze()
    labels = labels[~ignore]
    logits = logits[~ignore]

    return logits, labels


def get_metrics(logits, labels):
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
    err = torch.mean((preds != labels).float())
    return probs, err


def main():
    # Setup
    args = get_args()
    if args.track:
        wandb.init(project='ml_teaching', config=args)
    else:
        print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Model
    input_dimension = 0
    precomputed_dim = 0
    if 'pase' in args.features:
        input_dimension += PACE_EMB_DIM
    if 'mels' in args.features:
        input_dimension += MEL_DIM
        precomputed_dim += MEL_DIM
    if 'prosody' in args.features:
        input_dimension += PROSODY_DIM
        precomputed_dim += PROSODY_DIM

    if args.head == 'mlp':
        cls_head = CNNHead(input_dimension, args.classes, 1,
                           [args.hidden_size] * (args.hidden_layers - 1),
                           args.smooth, args.context_size, 1, args.norm, args.drop_hid)
    elif args.head == 'dtcnn':
        cls_head = CNNHead(input_dimension, args.classes, args.dilation_factor,
                           get_channel_progression(args.hidden_size, args.hidden_layers, update_rules=[1, 2]),
                           args.smooth, args.context_size, args.context_size, args.norm, args.drop_hid)
    elif args.head == 'lstm':
        cls_head = LSTMHead(input_dimension, args.classes, args.hidden_size, args.hidden_layers,
                            args.smooth, False, args.drop_hid)
    elif args.head == 'bilstm':
        cls_head = LSTMHead(input_dimension, args.classes, args.hidden_size//2, args.hidden_layers,
                            args.smooth, True, args.drop_hid)
    elif args.head == 'gru':
        cls_head = GRUHead(input_dimension, args.classes, args.hidden_size, args.hidden_layers,
                           args.smooth, False, args.drop_hid)
    elif args.head == 'bigru':
        cls_head = GRUHead(input_dimension, args.classes, args.hidden_size//2, args.hidden_layers,
                           args.smooth, True, args.drop_hid)

    model = PASEEncodedModel(cls_head, args.cfg, args.ckpt, drop_inp=args.drop_inp, drop_emb=args.drop_emb,
                             freeze_bn=args.freeze_bn, tune=args.tune).to(args.device)

    # Gradient accumulation
    torch.cuda.empty_cache()
    memory_avail = (torch.cuda.get_device_properties(args.device).total_memory - \
                    torch.cuda.memory_allocated(args.device)) / (1024**2)
    args.mb, n_acc_iters = get_accumulation_iters(model, (1, 160*args.seqlen), memory_avail, args.mb)

    # Optimizer
    param_groups = {'normal': [], 'no_decay': [], 'frozen': []}
    for n, p in model.named_parameters():
        if 'encoder' in n and not args.tune:
            param_groups['frozen'].append(p)
        elif 'act' in n:
            param_groups['no_decay'].append(p)
        else:
            param_groups['normal'].append(p)

    optimizer = torch.optim.AdamW([
            {'params': param_groups['normal'], 'weight_decay': args.wd},
            {'params': param_groups['no_decay']}
        ], lr=args.lr, amsgrad=args.amsgrad)

    samples_per_class = {
        4: [9528947, 2282939,  357391, 719512],
        5: [8831065,  698057, 2282939,  357391,  719512],
        9: [ 719512,  908666, 4733151,  683061, 2506712, 358743, 339489, 357391, 2282939.]
    }[args.classes]
    criterion = lambda x, y: cb_loss(x, y, samples_per_class, args.beta, args.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lrs_patience,
                                                           factor=args.lrs_factor)

    # Data
    with torch.no_grad():
        x = torch.rand(1, 1, 160*args.seqlen).to(args.device) if 'pase' in args.features else None
        p = torch.rand(1, precomputed_dim, args.seqlen).to(args.device)
        spacing = model(x, precomputed=p).detach().cpu().size(2)
        print(f'spacing={spacing}')

    trn_data = CAAMLRawFrameDataset('train', args.datapath, classes=args.classes, seq_len=args.seqlen,
                                    features=args.features, mask=args.mask, mask_mode=args.mask_mode,
                                    split_csv=args.split, spacing=spacing)
    dev_data = CAAMLRawFrameDataset('dev', args.datapath, classes=args.classes, seq_len=args.seqlen,
                                    features=args.features, split_csv=args.split, spacing=spacing)

    # Space out samples equivalent to outputted predictions
    trn_sampler = SpacedRandomSampler(trn_data, spacing=spacing)
    dev_sampler = SpacedSequentialSampler(dev_data, spacing=spacing)

    loader_args = {'batch_size': args.mb, 'num_workers': args.nworkers, 'pin_memory': True}
    trn_loader = torch.utils.data.DataLoader(trn_data, sampler=trn_sampler, **loader_args)
    dev_loader = torch.utils.data.DataLoader(dev_data, sampler=dev_sampler, **loader_args)

    # A bunch of stuff was run though the model before this, clear just to be safe.
    model.zero_grad()

    # Init metric values
    if args.track:
        wandb.watch(model)
    best_train_error, best_dev_error = 1, 1
    best_train_loss, best_dev_loss = float("inf"), float("inf")
    best_f1, best_mAP = 0, 0
    start_time = time.time()
    patience = args.patience
    for epoch in tqdm.tqdm(range(args.epochs), leave=True):

        # Train step
        model.train()
        trn_err, trn_loss = train(model, criterion, trn_loader, n_acc_iters, optimizer, args.clip_norm, args.device)

        # Log train metrics
        if args.track:
            if trn_err < best_train_error:
                best_train_error = trn_err
                wandb.run.summary['best_train_error'] = best_train_error
            if trn_loss < best_train_loss:
                best_train_loss = trn_loss
                wandb.run.summary['best_train_loss'] = best_train_loss

        # Eval step
        model.eval()
        dev_err, dev_loss, labels, probs = evaluate(model, criterion, dev_loader, n_acc_iters, args.device)

        # Log eval metrics
        f1 = f1_score(labels, np.argmax(probs, axis=1), average='macro')
        plots = CAAMLMetrics(probs, labels)
        if args.track:
            patience -= 1
            if best_dev_error - dev_err > 1e-4:
                best_dev_error = dev_err
                wandb.run.summary['best_dev_error'] = best_dev_error
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_best_err.pkl'))
            if best_dev_loss - dev_loss > 1e-4:
                best_dev_loss = dev_loss
                wandb.run.summary['best_dev_loss'] = best_dev_loss
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_best_loss.pkl'))
                patience = args.patience
            if f1 - best_f1 > 1e-4:
                best_f1 = f1
                wandb.run.summary['best_f1'] = f1
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_best_f1.pkl'))
            if plots.mAP - best_mAP > 1e-4:
                best_mAP = plots.mAP
                wandb.run.summary['best_mAP'] = best_mAP
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_best_mAP.pkl'))
            wandb.log({
                'metrics/train_error'    : trn_err,
                'metrics/train_loss'     : trn_loss,
                'metrics/dev_error'      : dev_err,
                'metrics/dev_loss'       : dev_loss,
                'metrics/f1'             : f1,
                'metrics/mAP'            : plots.mAP,
                'best/dev_error'         : best_dev_error,
                'best/dev_loss'          : best_dev_loss,
                'best/f1'                : best_f1,
                'best/mAP'               : best_mAP,
                'plots/precision-recall' : plots.prc_fig,
                'plots/confusion-matrix' : plots.cnf_fig,
                'plots/counts'           : plots.bar_fig
            })
        else:
            tqdm.tqdm.write(f'{epoch}: trn_err={trn_err:0.4f}, trn_loss={trn_loss:0.4f}, dev_err={dev_err:0.4f}, dev_loss={dev_loss:0.4f}, f1={f1:0.4f}')
        tqdm.tqdm.write(plots.report())

        # Step scheduler
        scheduler.step(dev_loss)

        if patience == 0 or (time.time() - start_time > args.timeout):
            break


if __name__ == "__main__":
    main()
