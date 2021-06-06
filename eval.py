#!/usr/bin/env python3

import argparse
import os
import pickle

import numpy as np
import plotly
from sklearn.metrics import f1_score

from metrics import CAAMLMetrics


plotly.io.orca.config.executable = '/research/hutchinson/bin/orca-1.3.1.AppImage'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.in_file, 'rb') as f:
        res = pickle.load(f)

    labs, probs = [], []
    for session in res.values():
        labs.append(session['labs'])
        probs.append(session['probs'])

    labs = np.concatenate(labs, axis=None)
    probs = np.concatenate(probs, axis=1).transpose(1, 0)
    preds = np.argmax(probs, axis=1)
    classes = probs.shape[1]

    # Plots
    plots = CAAMLMetrics(probs, labs)

    # Error
    per_class_err = np.zeros(classes)
    for c in range(classes):
        samples = labs == c
        per_class_err[c] = np.mean(preds[samples] != labs[samples])
    err_macro = np.mean(per_class_err)
    err_micro = np.mean(preds != labs)

    # F-Measures
    f_micro_r = f1_score(labs, preds,  average='micro')
    f_macro_r = f1_score(labs, preds,  average='macro')
    f_weighted_r = f1_score(labs, preds,  average='weighted')

    # Print and save output
    report = plots.report()
    my_report = f'err_micro={err_micro*100:02.2f}%, err_macro={err_macro*100:02.2f}%, mAP={plots.mAP:0.4f}, f1_micro={f_micro_r:0.4f}, f1_macro={f_macro_r:0.4f}, f1_weighted={f_weighted_r:0.4f}\n'
    print(my_report, end='')
    print(report)

    path = os.path.dirname(args.in_file)
    file = os.path.basename(args.in_file)
    file = '.'.join(file.split('.')[1:-1])
    plots.bar_fig.write_image(os.path.join(path, "bar_fig." + file + ".pdf"))
    plots.prc_fig.write_image(os.path.join(path, "prc_fig." + file + ".pdf"))
    plots.cnf_fig.write_image(os.path.join(path, "cnf_fig." + file + ".pdf"))
    with open(os.path.join(path, 'report.' + file + '.txt'), 'w') as f:
        f.write(my_report)
        f.write(report)


if __name__ == "__main__":
    main()
