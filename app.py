from flask import Flask

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from cycle_flow import *


server = Flask('cycle count', static_url_path='')
app = dash.Dash('Cycle count dashboard', server=server, url_base_pathname='/', csrf_protect=False)


# Hack to allow serving custom CSS. Taken from this:
# https://community.plot.ly/t/how-do-i-use-dash-to-add-local-css/4914/4
@server.route('/static/style.css')
def serve_stylesheet():
    return server.send_static_file('style.css')



data_df = pd.read_excel('/project/london_cycle_flow/tfl-cycle-flows-tlrn.xlsx', sheet_name=1)
clean_data_df = clean_data(data_df)
grid = create_grid()
lr = joblib.load('/project/london_cycle_flow/london-cycle-flow/model.pkl')

app.layout = html.Div(children=[
    html.Div(
        html.H1(children='Predicting the number of cycles in London')
    ),
    
    dcc.Input(
        placeholder='Total rainfall (mm)',
        id='rainfall-input',
        type='text',
        value=''
    ),
    
    dcc.Input(
        placeholder='Average temparature (C)',
        id='temperature-input',
        type='text',
        value=''
    ),
    
    html.Button('Predict', id='predict-button'),
    
    html.Div(id='params'),
    
    html.Div(id='plot', children=[
        dcc.Graph(id='counts-graph')
    ])
])


app.css.append_css({
    'external_url': '/static/style.css'
})
app.title = 'Cycle count dashboard'


@app.callback(
    Output(component_id='params', component_property='children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('rainfall-input', 'value'),
        State('temperature-input', 'value')]
)
def write_params(n_clicks, rainfall, temperature):
    if n_clicks and n_clicks>0:
        try:
            float(rainfall)
            float(temperature)
            
            new_div = html.Div(
                html.P('Parameters: {}, {}'.format(rainfall, temperature))
            )
            
        except:
            new_div = html.Div(children=[
                html.P('Parameters: {}, {}'.format(rainfall, temperature)),
                html.P(
                    'One or more non-numerical parameters: can\'t make prediction!',
                    style={'color': 'red'}
                )
            ]
            )
            
        
        return new_div
    


@app.callback(
    Output(component_id='counts-graph', component_property='figure'),
    [Input('predict-button', 'n_clicks')],
    [
        State('rainfall-input', 'value'),
        State('temperature-input', 'value')]
)
def plot_prediction(n_clicks, rainfall, temperature):
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
    
    if n_clicks > 0:
        try:
            rainfall = np.array([float(rainfall)])
            temperature = np.array([float(temperature)])
            
            trace3 = go.Scatter3d(
                x = rainfall,
                y = temperature,
                z = lr.predict(np.vstack((rainfall, temperature)).T).flatten(),
                mode = 'markers',
                marker=dict(
                    size=5,
                    color='red'
                ),
                name = 'new prediction'
            )
            
            data.append(trace3)
            
        except:
            pass

    layout = go.Layout(
        margin = dict(
            t = 50,
            b = 50
        ),
        width=800,
        height=600,
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