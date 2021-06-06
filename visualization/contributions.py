import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import mean_squared_error


with open('results_all/5/bigru/pase_mels_prosody/1/test1/predictions.raw.pkl', 'rb') as f:
    t1a = pickle.load(f)
with open('results_sanity/5/gru/mels/0/test1/predictions.raw.pkl', 'rb') as f:
    t1b = pickle.load(f)
with open('results_all/5/bigru/pase_mels_prosody/1/test2/predictions.raw.pkl', 'rb') as f:
    t2a = pickle.load(f)
with open('results_sanity/5/gru/mels/0/test2/predictions.raw.pkl', 'rb') as f:
    t2b = pickle.load(f)

t1 = {
    s: {
        'labs': t1a[s]['labs'],
        'probs_bst': t1a[s]['probs'],
        'probs_prv': t1b[s]['probs']
    } for s in t1a
}

t2 = {
    s: {
        'labs': t2a[s]['labs'],
        'probs_bst': t2a[s]['probs'],
        'probs_prv': t2b[s]['probs']
    } for s in t2a
}

float_formatter = "{:0.4f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
classes = next(iter(t1.values()))['probs_bst'].shape[0]
if classes == 9:
    labels = ['o ', 'a ', 'l ', 'iq', 'ia', 'sq', 'sa', 's ', 'g ']
    int2Label = {0: 'o', 1: 'a ', 2: 'l ', 3: 'iq', 4: 'ia', 5: 'sq', 6: 'sa', 7: 's ', 8: 'g '}
elif classes == 5:
    labels = ['ist', 'stu', 'grp', 'sil', 'oth']
    int2Label = {0: 'ist', 1: 'stu', 2: 'mlt', 3: 'sil', 4: 'oth'}
elif classes == 4:
    labels = ['sgl', 'mlt', 'sil', 'oth']
    int2Label = {0: 'sgl', 1: 'mlt', 2: 'sil', 3: 'oth'}

subplots = [[None]*2] * classes
fig, axs = plt.subplots(5, 2, figsize=[3.5, 3], dpi=300)

all_lab = []
all_raw = []
all_ens = []
for t in [t1, t2]:
    n_sessions = len(t)
    lab = np.zeros([n_sessions, classes])
    raw = np.zeros([n_sessions, classes])
    ens = np.zeros([n_sessions, classes])
    for idx, (session, data) in enumerate(t.items()):
        preds_raw = data['probs_bst'].argmax(axis=0)
        preds_ens = data['probs_prv'].argmax(axis=0)

        count_lab = dict(zip(*np.unique(data['labs'], return_counts=True)))
        count_raw = dict(zip(*np.unique(preds_raw, return_counts=True)))
        count_ens = dict(zip(*np.unique(preds_ens, return_counts=True)))

        count_lab = np.array([count_lab[c] if c in count_lab else 0 for c in range(classes)])
        count_raw = np.array([count_raw[c] if c in count_raw else 0 for c in range(classes)])
        count_ens = np.array([count_ens[c] if c in count_ens else 0 for c in range(classes)])
        assert sum(count_lab) == sum(count_raw) and sum(count_raw) == sum(count_ens)

        # Normalize to minutes
        # 100 frames / second & 60 seconds per minutes
        frames_to_minutes = 100 * 60
        count_lab = count_lab / frames_to_minutes
        count_raw = count_raw / frames_to_minutes
        count_ens = count_ens / frames_to_minutes

        lab[idx] = count_lab
        raw[idx] = count_raw
        ens[idx] = count_ens

    all_lab.append(lab)
    all_raw.append(raw)
    all_ens.append(ens)

for t in range(2):
    lab = all_lab[t]
    raw = all_raw[t]
    ens = all_ens[t]

    for c in range(classes):
        ax = axs[c][t]

        log_lab = lab[:, c]
        log_raw = raw[:, c]
        log_ens = ens[:, c]

        def apply_func(f, arrs):
            vals = [f(a) for a in arrs]
            return f(vals)

        if c == 1:
            print(log_lab)
            print(log_raw)
            print(log_ens)
            print('Min:', apply_func(np.min, [log_lab, log_raw, log_ens]))
            print('Max:', apply_func(np.max, [log_lab, log_raw, log_ens]))
            print()

        ax.scatter(log_lab, log_raw, marker='x', color="#ff6e00", s=16)
        ax.scatter(log_lab, log_ens, facecolors='none', edgecolors="black", s=16)
        ax.set_yscale('symlog', basey=10)
        ax.set_xscale('symlog', basex=10)

        # mse for minutes, r2 for percents
        raw_r2 = mean_squared_error(lab[:, c], raw[:, c], squared=False)
        ens_r2 = mean_squared_error(lab[:, c], ens[:, c], squared=False)

        # lim 0 min to 60 min
        lims = [0, 100]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax.tick_params(axis='both', which='major', labelsize=6, length=2)
        if t == 1:
            ax.set_yticklabels([])
        if c != classes - 1:
            ax.set_xticklabels([])
        ax.text(0.02, 0.95, f'X={raw_r2:0.2f} O={ens_r2:0.2f}', va='top', ha='left', transform=ax.transAxes, fontsize=6)
        if c == 0:
            ax.set_title(f'Test {t+1}', fontsize=7)
            ax.title.set_position([.5, 0.825])

for c in range(classes):
    axs[c][0].set_ylabel(labels[c], fontsize=7)

fig.subplots_adjust(hspace=0.275, wspace=0.1)
fig.savefig('contribution.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
