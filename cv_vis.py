
import numpy as np
import pandas as pd

# run the two lines in the notebook:
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)

import plotly.graph_objs as go
from plotly import tools

def plot_training_stats (logfile, log_loss=True):

    with open(logfile, 'r') as f:
        buff = f.readlines()
        # read the numbers of classes and figure out 'pure guess' accuracies
        i = 1; pure_guess = {}
        while i < len(buff) and buff[i] != '\n':
            label, cl, n_class = buff[i].split()
            pure_guess[cl] = float(n_class)
            i += 1
        ntotal = np.sum(list(pure_guess.values()))
        for key in pure_guess.keys():
            pure_guess[key] /= ntotal
        # read training stats
        df = pd.read_csv(logfile, header=i, sep='\t')
        del buff

    fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Loss', 'Accuracy'), print_grid=False)
    fig.append_trace(go.Scatter(x=df.Epoch, y=df.AvgTrainLoss, name='Train', line=dict(color='#1F77B4')), 1,1)
    fig.append_trace(go.Scatter(x=df.Epoch, y=df.AvgValLoss, name='Val', line=dict(color='#FF7F0E')), 1,1)
    fig.append_trace(go.Scatter(x=df.Epoch, y=df.AvgTrainAcc, name='Train', showlegend=False, line=dict(color='#1F77B4')), 2,1)
    fig.append_trace(go.Scatter(x=df.Epoch, y=df.AvgValAcc, name='Val', showlegend=False, line=dict(color='#FF7F0E')), 2,1)
    fig['layout']['xaxis2'].update(title='Epoch')
    if log_loss:
        fig['layout']['yaxis1'].update(type='log')

    for cl in pure_guess.keys():
        fig.append_trace(go.Scatter(x=df.Epoch, y=(pure_guess[cl]*np.ones(len(df.Epoch))), mode='lines', name=('Always %s' % cl), showlegend=False, line=dict(color='#000000', dash='dot')), 2,1)

    return fig