#! /usr/bin/env python3
import argparse
from itertools import groupby
import os
import pickle
import time

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.core.utils.types import Label, LabelStyle
import tqdm


# Forked Notebook from pyannote.core and heavily modified so that each
# track is plotted on it's own line. Each track essintally becomes a
# different concurrent hypothesis.
COLORS = ["#0a4f4e", "#5dd12f", "#450054", "#fe74fe", "#6b14d6", "#dce31b", "#abc177", "#458612", "#565bd9"]


class Notebook:
    def __init__(self):
        self.reset()

    def reset(self):
        del self.crop
        del self.width

    @property
    def crop(self):
        """The crop property."""
        return self._crop

    @crop.setter
    def crop(self, segment: Segment):
        self._crop = segment

    @crop.deleter
    def crop(self):
        self._crop = None

    @property
    def width(self):
        """The width property"""
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @width.deleter
    def width(self):
        self._width = 20

    def __getitem__(self, label: Label) -> LabelStyle:
        if label not in self._style:
            self._style[label] = next(self._style_generator)
        return self._style[label]

    def setup(self, ax=None, ylim=(0, 1), yaxis=False, ttime=True):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(self.crop)
        if ttime:
            ax.set_xlabel('Elapsed Time (MM:SS)', fontsize=6, labelpad=2)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(ylim)
        ax.axes.get_yaxis().set_visible(yaxis)
        return ax

    def draw_segment(self, ax, segment: Segment, y, color_map=None, label=None, boundaries=False):

        # do nothing if segment is empty
        if not segment:
            return
        color = color_map[label]
        linewidth = (7.16/16)*15

        # draw segment
        ax.hlines(y, segment.start, segment.end, color,
                  linewidth=linewidth, label=label)
        if boundaries:
            ax.vlines(segment.start, y + 0.05, y - 0.05,
                      color, linewidth=1, linestyle='solid')
            ax.vlines(segment.end, y + 0.05, y - 0.05,
                      color, linewidth=1, linestyle='solid')

    def plot_annotation(self, labels, preds, color=None, ax=None, ttime=True, legend=True):

        if not self.crop:
            self.crop = labels.get_timeline(copy=False).extent()

        cropped_labels = labels.crop(self.crop, mode='intersection')
        cropped_preds = preds.crop(self.crop, mode='intersection')
        labels = cropped_labels.labels()
        ax = self.setup(ax=ax, ttime=time)

        msegment = 0
        for (segment, track, label) in cropped_labels.itertracks(yield_label=True):
            self.draw_segment(ax, segment, 3/4 - (1/4/8), color_map=color, label=label)
            msegment = max(msegment, segment.end)

        for (segment, track, label) in cropped_preds.itertracks(yield_label=True):
            self.draw_segment(ax, segment, 1/4 + (1/4/8), color_map=color, label=label)

        if legend:
            H, L = ax.get_legend_handles_labels()
            if not H:
                return

            # this gets exactly one legend handle and one legend label per label
            # (avoids repeated legends for repeated tracks with same label)
            order = {'instructor': 0, 'student': 1, 'group': 2, 'silence': 3, 'other': 4}
            HL = groupby(sorted(zip(H, L), key=lambda h_l: order[h_l[1]]),
                         key=lambda h_l: h_l[1])
            H, L = zip(*list((next(h_l)[0], l) for l, h_l in HL))

            ax.legend(H, L, bbox_to_anchor=(0, 0.85), loc=3,
                      ncol=5, borderaxespad=0., frameon=False, fontsize=6)

        formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
        ax.xaxis.set_major_formatter(formatter)

        locator = matplotlib.ticker.LinearLocator(numticks=6)
        ax.xaxis.set_major_locator(locator)


def main():
    # Setup
    args = get_args()

    with open(args.in_file, 'rb') as f:
        res = pickle.load(f)
    fig_root = os.path.join(os.path.dirname(args.in_file), 'traces/')
    if not os.path.exists(fig_root):
        os.mkdir(fig_root)

    classes = next(iter(res.values()))['probs'].shape[0]
    if classes == 9:
        lab2str = {0: 'o', 1: 'a ', 2: 'l ', 3: 'iq', 4: 'ia', 5: 'sq', 6: 'sa', 7: 's ', 8: 'g '}
        colors1 = COLORS
    elif classes == 5:
        lab2str = {0: 'instructor', 1: 'student', 2: 'group', 3: 'silence', 4: 'other'}
        colors1 = ["#ff6e00", "#03c991", "#4aaee8", "#d590c8", "#016398"]
    elif classes == 4:
        lab2str = {0: 'single-voice', 1: 'multi-voice', 2: 'silence', 3: 'other'}
        colors1 = [COLORS[2], COLORS[8], COLORS[7], COLORS[0]]
    str2clr = {lab2str[i]: colors1[i] for i in range(len(colors1))}

    labs = []
    for session, data in tqdm.tqdm(res.items()):
        labs = data['labs']
        preds = np.argmax(data['probs'], axis=0)

        # Read annotation files
        labels = Annotation(uri=session, modality="speaker")
        for idx, l in enumerate(labs):
            labels[Segment(idx*0.01, (idx+1)*0.01), f'L{idx}'] = lab2str[l]

        predictions = Annotation(uri=session, modality="speaker")
        for idx, l in enumerate(preds):
            predictions[Segment(idx*0.01, (idx+1)*0.01), f'P{idx}'] = lab2str[l]

        labels = labels.support(collar=0.0)
        predictions = predictions.support(collar=0.0)

        # Create two separate subplots, use figsize to control aspect ratio
        fig, axs = plt.subplots(figsize=((7.16/16)*16, (7.16/16)*0.8))

        # Plot the two annotation with pyannote.metrics
        notebook = Notebook()
        notebook.plot_annotation(labels, predictions, color=str2clr, ax=axs, ttime=True, legend=True)

        axs.annotate("Ground Truth",
                     xy=((7.16/16)*10, 3/4 - (1/4/8) - 0.09), xycoords='data', color='white', size=6,
                     path_effects=[PathEffects.withStroke(linewidth=(7.16/16)*1, foreground="black")])

        axs.annotate("Predicted",
                     xy=((7.16/16)*10, 1/4 + (1/4/8) - 0.09), xycoords='data', color='white', size=6,
                     path_effects=[PathEffects.withStroke(linewidth=(7.16/16)*1, foreground="black")])

        # Save the plot in svg format
        axs.spines['left'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        fig_path = os.path.join(fig_root, f'{session}.pdf')
        plt.xticks(fontsize=6)
        fig.savefig(fig_path, format='pdf', bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
