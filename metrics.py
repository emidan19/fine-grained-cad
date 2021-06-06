#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize


# plotly.io.orca.config.executable = '/usr/bin/orca' --- change to your orca path if necessary


'''
    Welcome to CAAMLMetrics2.0

    Everything is the same except it's completely different and way better now.
    Your same exact inputs to CAAMLMetrics1.0 should still work for CAAMLMetrics2.0,
    but the way you get the figures is going to be different and requires some setup.

----SETUP----
    To use plotly and to save the figures you'll need some packages-
    pip3/conda install plotly
    pip3/conda install psutil
    pip3/conda install requests

    conda install -c plotly plotly-orca

    Supposedly the pip version doesn't work. The conda version didn't work for me either.
    If your install for orca doesn't work you can try this-
    set the 'plotly.io.orca.config.executable' property to the full path of your orca exe
    in your CAAMLMetrics file. Just change the path in the #commented line above this comment block.

----INPUTS----
    Input to CAAMLMetrics2 object-
    probs  -  [# frames][C]
    labs   -  [# frames][C]

----HOW-TO----
    4-way vs 9-way is inferred based on the lists you pass in.
    The second dimension of probs/labs should be your # classes.
    C == len(probs[0])

    Instantiate a CAAMLMetrics object at the end of your evaluation like before.
    m = CAAMLMetrics(probs, labs)

----OUTPUT----
    Call a function if you want to print the report.
    m.report()

    Use handles to get figures. Save them where and how you want to.
    bar_fig = m.bar_fig
    prc_fig = m.prc_fig
    cnf_fig = m.cnf_fig

    To save a static image of plotly figure to /images-
    from plotly.io import write_image
    bar_fig.write_image("images/bar_fig_static.svg")

    To save an interactive plotly figure to an html file in /html-
    from plotly.io import write_html
    bar_fig.write_html("html/bar_fig_interact.html")
'''

COLORSA = ['#cc0000', '#ff6666', '#ff6633', '#ffcc00', '#009900', '#66ff66', '#00cccc', '#0033ff', '#660099']
COLORSB = ['#aa0000', '#dd4444', '#dd4411', '#ddaa00', '#007700', '#44dd44', '#00aaaa', '#0011dd', '#440077']


class CAAMLMetrics:

    def __init__(self, probs, labs, preds=None):
        C = len(probs[0])
        labs_ext = np.array(label_binarize(labs, classes=range(0, C)))  # Labels extended to [#frames][C]
        probs_np = np.array(probs)
        if preds is None:
            preds = [np.argmax(row) for row in probs_np]  # Prediction labels of probs

        if C == 9:
            labels = ['o', 'a', 'l', 'iq', 'ia', 'sq', 'sa', 's', 'g']
            colors1 = COLORSA
            colors2 = COLORSB
        elif C == 5:
            colors1 = ["#ff6e00", "#03c991", "#4aaee8", "#d590c8", "#016398"]
            colors2 = ["#ff6e00", "#03c991", "#4aaee8", "#d590c8", "#016398"]
            labels = ['ist', 'stu', 'grp', 's', 'o']
        elif C == 4:
            colors1 = [COLORSA[2], COLORSA[8], COLORSA[7], COLORSA[0]]
            colors2 = [COLORSB[2], COLORSB[8], COLORSB[7], COLORSB[0]]
            labels = ['sgl', 'mlt', 'sil', 'oth']

        self.labs = labs
        self.labs_ext = labs_ext
        self.preds = preds
        self.probs_np = probs_np
        self.probs = probs
        self.labels = labels
        self.colors1 = colors1
        self.colors2 = colors2
        self.C = C
        self.bar_fig = self.bar_chart()
        self.prc_fig, self.mAP = self.pr_curves()
        self.cnf_fig = self.cm()

    def report(self):
        preds = self.preds
        labs = self.labs
        labels = self.labels
        return classification_report(labs, preds, target_names=labels)

    def bar_chart(self):
        labels = self.labels
        labs = self.labs
        preds = self.preds
        colors1 = self.colors1
        colors2 = self.colors2
        C = self.C

        true_count = [0]*C
        pred_count = [0]*C

        for i in range(len(labs)):
            true_count[labs[i]] += 1
            pred_count[preds[i]] += 1

        total = sum(true_count)

        true_bar = []
        pred_bar = []

        for i in range(C):
            true_bar.append(int((100*true_count[i] / total) * 10) / 10)
            pred_bar.append(int((100*pred_count[i] / total) * 10) / 10)

        # create df from bars
        bars_df = pd.DataFrame(true_bar, columns=['true'])
        bars_df['pred'] = pred_bar

        true_per = []
        pred_per = []

        for i in range(C):
            true_per.append(str(true_count[i]*100/total)+'%')
            pred_per.append(str(pred_count[i]*100/total)+'%')

        trace1 = go.Bar(x=labels, y=pred_bar, name='Predicted', marker_color=colors1, text=pred_per)
        trace2 = go.Bar(x=labels, y=true_bar, name='Actual', marker_color=colors2, text=true_per)

        data = [trace1, trace2]

        layout = go.Layout(barmode='group', showlegend=False, title="Predictions (left) vs Actual Labels (right)")
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0))
        return fig

    def pr_curves(self):
        labels = self.labels
        labs_ext = self.labs_ext
        probs_np = self.probs_np
        colors1 = self.colors1
        colors2 = self.colors2
        C = self.C

        precision, recall, ap = [0]*C, [0]*C, [0]*C
        for i in range(C):
            precision[i], recall[i], _ = precision_recall_curve(labs_ext[:, i], probs_np[:, i])
            ap[i] = average_precision_score(labs_ext[:, i], probs_np[:, i])

        fig = go.Figure()
        for i in range(C):
            fig.add_trace(go.Scatter(x=recall[i][::1000], y=precision[i][::1000], mode='lines', name=labels[i],
                          line_color=colors1[i], text=f'aoc: {ap[i]:0.2f}'))

        mAP = sum(ap) / len(ap)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0))

        fig.update_layout(legend=dict(yanchor="bottom", y=0.025, xanchor="left", x=0.025))
        fig.update_xaxes(tickfont=dict(size=18, color='black'))
        fig.update_yaxes(tickfont=dict(size=18, color='black'))
        fig.layout.legend.font.size = 18
        fig.layout.legend.font.color = 'black'
        return fig, mAP

    def cm(self):
        z = confusion_matrix(self.labs, self.preds, normalize='true')[::-1]
        z_text = [[f'{y:.2f}'[1:] for y in x] for x in z]

        fig = ff.create_annotated_heatmap(z, zmin=0.0, zmax=1.0, x=self.labels, y=self.labels[::-1],
                                          annotation_text=z_text, colorscale='viridis', showscale=True)
        fig['data'][0]['colorbar.tickfont'] = dict(size=44, color='black')
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0))
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 44
        fig.update_xaxes(tickfont=dict(size=44, color='black'))
        fig.update_yaxes(tickfont=dict(size=44, color='black'))
        return fig
