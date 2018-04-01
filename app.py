from flask import Flask

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from cycle_flow import *


server = Flask('cycle count')
app = dash.Dash('Cycle count dashboard', server=server, url_base_pathname='/', csrf_protect=False)



data_df = pd.read_excel('/project/london_cycle_flow/tfl-cycle-flows-tlrn.xlsx', sheet_name=1)
clean_data_df = clean_data(data_df)
grid = create_grid()
lr = joblib.load('/project/london_cycle_flow/london-cycle-flow/model.pkl')

app.layout = html.Div(children=[
    html.Div(
        html.H1(children='Cycle counts')
    ),
    
    dcc.Input(
        placeholder='Total rainfall (mm)...',
        id='rainfall-input',
        type='text',
        value=''
    ),
    
    dcc.Input(
        placeholder='Average temparature (C)...',
        id='temperature-input',
        type='text',
        value=''
    ),
    
    html.Button('Predict', id='predict-button'),
    
    html.Div(id='plot', children=[
        dcc.Graph(id='counts-graph')
    ])
])




@app.callback(
    Output(component_id='counts-graph', component_property='figure'),
    [
        Input('rainfall-input', 'value'),
        Input('temperature-input', 'value'),
        Input('predict-button', 'n_clicks')
    ]
)
def plot_prediction(rainfall, temperature, n_clicks=0):
    if not n_clicks:
        n_clicks = 0
    
    trace1 = go.Scatter3d(
        x = clean_data_df['total_rainfall_mm'],
        y = clean_data_df['avg_temp_c'],
        z = clean_data_df['cycle_counts'],
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
    
    if rainfall and temperature:
        rainfall = float(rainfall)
        temperature = float(temperature)
        
        trace3 = go.Scatter3d(
            x = rainfall,
            y = temperature,
            z = lr.predict(np.array([rainfall, temperature]).reshape((1,2))),
            mode = 'markers',
            marker=dict(
                size=5,
                color='red'
            ),
            name = 'new prediction'
        )
        
        data.append(trace3)

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
    
    return fig
        

        

if __name__ == '__main__':
    #app.server.run(host='127.0.0.1', port=8888)
    app.server.run(port=8888, debug=True)