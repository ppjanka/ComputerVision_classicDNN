
import numpy as np
import pandas as pd

# run the two lines in the notebook:
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)

import plotly.graph_objs as go
from plotly import tools

def plot_training_stats (logfile):
    df = pd.read_csv(logfile, sep='\t')

    fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Loss', 'Accuracy'), print_grid=False)
    fig.append_trace(go.Scatter(x=df.Epoch, y=df.AvgTrainLoss, name='Train', line=dict(color='#1F77B4')), 1,1)
    fig.append_trace(go.Scatter(x=df.Epoch, y=df.AvgValLoss, name='Val', line=dict(color='#FF7F0E')), 1,1)
    fig.append_trace(go.Scatter(x=df.Epoch, y=df.AvgTrainAcc, name='Train', showlegend=False, line=dict(color='#1F77B4')), 2,1)
    fig.append_trace(go.Scatter(x=df.Epoch, y=df.AvgValAcc, name='Val', showlegend=False, line=dict(color='#FF7F0E')), 2,1)
    fig['layout']['xaxis2'].update(title='Epoch')

    return fig