import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

def clean_data(data_df):
    renaming = {
        'level_1': 'start',
        'level_2': 'end',
        'level_3': 'cycle_counts',
        'Unnamed: 5': 'avg_temp_c',
        'Unnamed: 7': 'total_rainfall_mm',
        'Unnamed: 8': 'avg_wet_hrs_per_day',
    }

    to_drop = [
        'level_0',
        'level_4',
        'Comparisons analysis',
        'Unnamed: 1',
        'Unnamed: 2',
        'Unnamed: 3',
        'Unnamed: 4',
        'Unnamed: 6'
    ]
    
    clean_data = data_df.reset_index().drop(['level_5'], axis=1).dropna()[1:]
    
    clean_data = (clean_data.rename(renaming, axis=1).drop(to_drop, axis=1).reset_index()).drop('index', axis=1)
    #cleaner_data = cleaner_data.drop('index', axis=1)
    
    return clean_data
    
    
def plot2d(clean_data_df):
    trace1 = go.Scatter(
        x = clean_data_df['total_rainfall_mm'],
        y = clean_data_df['cycle_counts'],
        mode = 'markers'
    )

    trace2 = go.Scatter(
        x = clean_data_df['avg_temp_c'],
        y = clean_data_df['cycle_counts'],
        mode = 'markers'
    )

    fig = tools.make_subplots(rows=1, cols=2)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout'].update(height=600, width=1200)

    fig['layout']['xaxis1'].update(title='rainfall (mm)')
    fig['layout']['yaxis1'].update(title='cycle counts')

    fig['layout']['xaxis2'].update(title='average temperature (C)')
    fig['layout']['yaxis2'].update(title='cycle counts')

    iplot(fig)
    
    
def plot3d(clean_data_df):
    trace = go.Scatter3d(
        x = clean_data_df['total_rainfall_mm'],
        y = clean_data_df['avg_temp_c'],
        z = clean_data_df['cycle_counts'],
        mode = 'markers',
        marker = dict(
            size = 5
        )
    )

    data = [trace]

    layout = go.Layout(
        scene = dict(
            xaxis = dict(
                title = 'total rainfall (mm)'
            ),
            yaxis = dict(
                title = 'average temperature (C)'
            ),
            zaxis = dict(
                title = 'cycle counts'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
    
    
def plot_seasonality(clean_data_df):
    trace = go.Scatter(
        x = clean_data_df['start'],
        y = clean_data_df['cycle_counts']
    )
    
    layout = go.Layout(
        xaxis = dict(
            title = 'date (period start)'
        ),
        yaxis = dict(
            title = 'cycle count'
        ),
        title = 'Time series'
    )
    
    data = [trace]
    
    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
    
    
def get_features_targets(clean_data_df):
    X = clean_data_df[['total_rainfall_mm', 'avg_temp_c']]

    Y = np.array(clean_data_df['cycle_counts'])
    Y = Y.reshape((len(Y),1))
    
    return X, Y
    
    
def create_grid():
    xc = np.linspace(0, 120, 100)
    yc = np.linspace(0, 20, 100)
    
    xgrid, ygrid = np.meshgrid(xc, yc)
    
    grid = np.vstack((xgrid.flatten(),ygrid.flatten())).T
    
    return grid
    
    
def plot_predictions(clean_data_df, grid, lr):
    trace1 = go.Scatter3d(
        x = cleaner_data['total_rainfall_mm'],
        y = cleaner_data['avg_temp_c'],
        z = cleaner_data['cycle_counts'],
        mode = 'markers',
        marker=dict(
            size=5
        ),
        name = 'data'
    )

    trace2 = go.Scatter3d(
        x = grid[:,0],
        y = grid[:,1],
        z = lr.predict(grid).flatten(),
        mode = 'markers',
        marker=dict(
            size=1,
            color='rgb(255,255,102)',
            opacity=0.6
        ),
        name = 'prediction'
    )

    data = [trace1, trace2]

    layout = go.Layout(
        margin = dict(
            t = 50,
            b = 50
        ),
        scene = dict(
            xaxis = dict(
                title = 'total rainfall (mm)'
            ),
            yaxis = dict(
                title = 'average temperature (C)'
            ),
            zaxis = dict(
                title = 'cycle counts'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)