from dash import Input, Output, State, dcc, html, Dash
#from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from Business_layer import GraphBuilder
from dash_bootstrap_templates import load_figure_template
load_figure_template("MORPH")

graph=GraphBuilder()
static=graph.plot_source_geometry()
velocity=graph.plot_vs_vp_profiles()
shot=graph.plot_shot_gather_XD()
dbc_css="https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__,external_stylesheets=[dbc.themes.MORPH,dbc_css])

app.layout = dbc.Container([
    html.H1("Graphs", className="text-center"),  

    dbc.Row([
        dbc.Col(dcc.Graph(figure=static), width=4),
        dbc.Col(dcc.Graph(figure=velocity), width=4),
        dbc.Col([dcc.Graph(figure=shot),
                 html.H3("Amplitude"),
                 dcc.Slider(min=0,max=100,step=20,value=20,id="amplitude-slider",className="dbc"),
                 html.H3("Time"),
                 dcc.Slider(min=0,max=20,step=1,value=5,id="time-slider",className="dbc")
                
                
                ]),
    ]),
    html.H3("Model"),
    dcc.Slider(min=0,max=11,step=1,value=1,id="model-number-slider",className="dbc"), 
    html.H3("MT Source (0-isotropic, 1-Mxx, 2-Myy, 3-Mzz, 4-Mxy)"),
    dcc.Slider(min=0,max=4,step=1,value=0,id="'mt-source-slider",className="dbc"), 
    html.H3("Source Angle (0-red, 1-blue, 2-yellow, 3-cyan, 4-Magenta)"),
    dcc.Slider(min=0,max=4,step=1,value=0,id="'source-angle-slider",className="dbc")
], fluid=True)
    
        