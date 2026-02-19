import dash
from dash import dcc, html, ctx
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import MATCH, ALL, Patch
from dash import dash_table
import plotly.graph_objects as go
import pickle
from sklearn.tree import DecisionTreeClassifier
# from sklearn.dummy import DummyClassifier
import math
from sklearn.model_selection import cross_val_score, cross_validate, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, fisher_exact, ks_2samp, ttest_ind, mannwhitneyu
from statsmodels.stats.contingency_tables import Table
from datetime import date, datetime
# from sklearn.metrics import roc_auc_score
import pyodbc
from sqlalchemy import create_engine, text
from urllib import parse
# import time
# import tracemalloc
# import ipg_functions as ipg
# from xgboost import XGBRegressor
import xgboost as xgb
import base64
import io
import dash_ag_grid as dag
from pandas.api.types import is_numeric_dtype


# app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, "style.css"])
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


# ############# Input Data and Model #######################################
# with open("fake_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "Salary"

# with open("fake_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False



############# Input Data and Model #######################################
with open("bob_data2.pickle", "rb") as file:
    df = pickle.load(file)
print(df.shape)
target = "is_good"

with open("bob_model2.pickle", "rb") as file:
    model = pickle.load(file)
is_classification = True

# ############# Input Data and Model #######################################
# with open("bob_data_90.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "is_good"

# with open("bob_model_90.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = True

# ############# Input Data and Model #######################################
# with open("bob_data_regressor.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "JJ Active Processing Speed (Feet/Minute)"

# with open("bob_model_regressor.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False



# ############# Input Data and Model #######################################
# with open("die_pressure_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "HMC Die Pressure Act Mean (PSI)"

# with open("die_pressure_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


# ############# Input Data and Model #######################################
# with open("die_pressure_visc_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "HMC Die Pressure Act Mean (PSI)"

# with open("die_pressure_visc_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


# ############# Input Data and Model #######################################
# with open("coated_scanner_spread_visc_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "Coated Scanner Spread (%)"

# with open("coated_scanner_spread_visc_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


# ############# Input Data and Model #######################################
# with open("yield_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "Slitter Yield (%)"

# with open("yield_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


# ############# Input Data and Model #######################################
# with open("die_pressure_data2.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "Die Pressure Act  (PSI)"

# with open("die_pressure_model2.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


############# Input Data and Model #######################################
# with open("HMC_Jan2026_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "Die Pressure Act  (PSI)"

# with open("HMC_Jan2026_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False



# ############# Get Max Contribution ######################################
# if is_classification:
#     max_contrib = 0.5
# else:
#     max_contrib = df[target].std() * 2
# y = df[target]
# print(max_contrib)


# ############# Get Predicted value precision ##############################
# prediction_std = df[target].std()
# if prediction_std < 0.1:
#     sig_figs = 3
# elif prediction_std < 1:
#     sig_figs = 2
# else:
#     sig_figs = 1


# ############# Drop Target ################################################
# df = df.drop(columns=target)


# ############## Get Categorical Features and values ###################################
# categorical_features = {}
# categorical_df = df.select_dtypes(include=["object"])
# for categorical_feature in categorical_df.columns:
#     categorical_features[categorical_feature] = list(np.unique(categorical_df[categorical_feature].dropna()))


# ############## Get model features #########################################
# model_features = list(pd.get_dummies(df, prefix_sep="?").columns)



def sort_features_on_importance(model, X):
    # Encode df
    encoded_df = pd.get_dummies(X, prefix_sep="?")

    # Get prediction contributions
    contribs = model.get_booster().predict(xgb.DMatrix(encoded_df), pred_contribs=True)[:, :-1]
    
    # Store prediction contributions in a dataframe
    contrib_df = pd.DataFrame(contribs, columns=encoded_df.columns)
    # print(contrib_df)

    # Aggregate SHAP values for categorical features
    contrib_df = contrib_df.groupby(lambda x: x.split('?')[0], axis=1).sum()

    # Get maximum contribution
    # some_max = contrib_df.abs().max().max()
    # max_feature = contrib_df.columns[contrib_df.max().argmax()]
    # min_feature = contrib_df.columns[contrib_df.min().argmin()]
    # print(max_feature)
    # print(min_feature)
    
    # Place contributions in a dataframe, calculate avg positive contribution score, and sort
    temp = pd.DataFrame({"features":contrib_df.columns, "avg_abs_contribs":contrib_df.abs().mean(axis=0)}).sort_values("avg_abs_contribs", ascending=False)
    # temp = pd.DataFrame({"features":contrib_df.columns, "std_contribs":contrib_df.std(axis=0)}).sort_values("std_contribs", ascending=False)
    # temp = pd.DataFrame({"features":contrib_df.columns, "std_contribs":contrib_df.abs().max(axis=0)}).sort_values("std_contribs", ascending=False)
    # temp = pd.DataFrame({"features":contrib_df.columns, "std_contribs":contrib_df.max(axis=0)-contrib_df.min(axis=0)}).sort_values("std_contribs", ascending=False)
    
    return list(temp["features"])

# ############### Get Features list sorted by contribution ###############
# ordered_features = sort_features_on_importance(model, df)



def update_model_metadata(model, df_clean_target):
    global max_contrib
    global sig_figs
    global categorical_features
    global model_features
    global ordered_features

    ############# Get Max Contribution ######################################
    if is_classification:
        max_contrib = 0.5
    else:
        max_contrib = df_clean_target[target].std() * 2
    print(max_contrib)

    ############# Get Predicted value precision ##############################
    prediction_std = df_clean_target[target].std()
    if prediction_std < 0.1:
        sig_figs = 3
    elif prediction_std < 1:
        sig_figs = 2
    else:
        sig_figs = 1
    
    ############## Get Categorical Features and values ###################################
    X = df_clean_target.drop(columns=target)
    categorical_features = {}
    categorical_df = X.select_dtypes(include=["object"])
    for categorical_feature in categorical_df.columns:
        categorical_features[categorical_feature] = list(np.unique(categorical_df[categorical_feature].dropna()))

    ############## Get model features #########################################
    model_features = list(pd.get_dummies(X, prefix_sep="?").columns)

    ############### Get Features list sorted by contribution ###############
    ordered_features = sort_features_on_importance(model, X)
    

update_model_metadata(model, df)
if is_classification:
    unique_vals = np.unique(df[target].dropna())
    target_map = {0:unique_vals[0], 1:unique_vals[1]}





def get_indicator_KPI(curr_value, prev_value, title):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        value=curr_value,
        number={'font': {'size': 40}},
        delta = {'reference': prev_value, "font":{"size":14}},
        title={"text":title, "font":{"size":20}},
        gauge={
            'axis': {'range': [0, 100], 'visible': True}},
        # domain = {'row': 0, 'column': 0},
        # number = {'suffix': "%"}
    ))
    fig.update_layout(
        template = {'data' : {'indicator': [{
            'mode' : "number+delta+gauge"}]
                            }},
        height=180,
        width=280,
        margin=dict(l=30, r=35, t=50, b=0)
    )
    return fig


app.layout = html.Div([

    html.Button("click me", id="update-page-btn", n_clicks=0, style={"border":"none","color":"transparent"}),
    html.Div(
        # target.upper() + " - SIMULATION",
        id = "page-title",
        style={"position":"absolute","height":"8vh","top":"2vh","left":"5vw","fontSize":"4vh","fontWeight":"600"}),

    # html.Div(
    #     id="feature-slider-container",
    #     style={"position":"absolute","height":"80vh","width":"15vw","left":"10vw","top":"15vh","border":"1px solid",
    #            "overflowY":"auto","overflowX":"hidden"}),

    # html.Div(
    #     id="positive-contribution-container",
    #     style={"position":"absolute","height":"80vh","width":"15vw","left":"30vw","top":"15vh","border":"1px solid",
    #            "overflowY":"auto","overflowX":"hidden"}),

    html.Div([
        
        html.Div(
            id="feature-slider-container",
            style={"position":"absolute","height":"80vh","width":"15vw","left":"15vw","top":"40px","border":"0px solid"}),#"top":"5vh"

        html.Div(
            id="positive-contribution-container",
            style={"position":"absolute","height":"80vh","width":"15vw","left":"30vw","top":"40px","border":"0px solid"}),#"top":"5vh"

        html.Div(
            id="negative-contribution-container",
            style={"position":"absolute","height":"80vh","width":"15vw","left":"0vw","top":"40px","border":"0px solid"}),#"top":"5vh"


    ], style={"position":"absolute","height":"80vh","width":"45vw","left":"5vw","top":"15vh","border":"0px solid",
              "overflowY":"auto","overflowX":"hidden","boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)",
              "backgroundColor":"#ffffff"}, className="split-background"),#"backgroundColor":"#fff"
    
    html.Div([
        "CONTRIBUTIONS BY TAG"
    ], style={"position":"absolute","height":"4vh","width":"45vw","left":"5vw","top":"11vh","fontSize":"2.5vh","border":"0px solid",
              "borderTopLeftRadius":"5px","borderTopRightRadius":"5px","backgroundColor":"#dfdfdf","paddingLeft":"1vw",
              "color":"#599ad3","fontWeight":"600","boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)",
              "clipPath":"inset(0px -5px 0px -5px)"}),

    

    html.Div([
        # html.Div([
        #     html.Div([
        #         html.Span("57.3", style={"border":"0px solid","lineHeight":"1"}), 
        #         html.Span("%",style={"border":"0px solid","fontSize":"2.5vh","marginBottom":"1vh"}),
        #     ], style={"width":"100%","border":"0px solid","marginTop":"1vh"}),
            
        #     html.Div("Good Run", style={"border":"0px solid","fontSize":"2vh","lineHeight":"1","paddingBottom":"1vh"}),
        # ], style={"position":"relative","width":"8vw","float":"left","border":"0px solid","textAlign":"center","color":"#333333"}),


        # html.Div([
        #     html.Div([
        #         html.Span("62.9", style={"border":"0px solid","lineHeight":"1"}), 
        #         html.Span("%",style={"border":"0px solid","fontSize":"2.5vh","marginBottom":"1vh"}),
        #     ], style={"width":"100%","border":"0px solid","marginTop":"1vh"}),
            
        #     html.Div("Bad Run", style={"border":"0px solid","fontSize":"2vh","lineHeight":"1","paddingBottom":"1vh"}),
        # ], style={"position":"relative","width":"8vw","float":"right","border":"0px solid","textAlign":"center","color":"#333333"}),


        html.Div([
            html.Div([
                html.Span("357.3", style={"border":"0px solid","lineHeight":"1"}), 
            ], style={"width":"100%","border":"0px solid","marginTop":"1vh"}),
            
            html.Div("Die Pressure Act", style={"border":"0px solid","fontSize":"2vh","lineHeight":"1","paddingBottom":"1vh"}),
        ], style={"position":"relative","width":"100%","border":"0px solid","textAlign":"center","color":"#333333"}),


        
    ], id="model-prediction",
    style={"position":"absolute","width":"12vw","left":"55vw","top":"15vh","border":"0px solid","fontSize":"5vh",
           "boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)","backgroundColor":"#ffffff",
           "color":"#757575","fontWeight":"600"}),

    
    html.Div([
        "ESTIMATED"
    ], style={"position":"absolute","height":"4vh","width":"12vw","left":"55vw","top":"11vh","fontSize":"2.5vh","border":"0px solid",
              "borderTopLeftRadius":"5px","borderTopRightRadius":"5px","backgroundColor":"#dfdfdf","paddingLeft":"1vw",
              "color":"#599ad3","fontWeight":"600","boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)",
              "clipPath":"inset(0px -5px 0px -5px)"}),


    dbc.Modal([
        # dbc.ModalHeader(dbc.ModalTitle("", id="modal-title")),
        dbc.ModalBody([
            
            dcc.Graph(
                id="contrib-graph",
                style={"position":"relative","height":"50vh","left":"0%","width":"100%","top":"0vh","border":"0px solid","padding":"5px", 
                    "boxSizing":"border-box"},#"height":"50vh","left":"0%","width":"60vw"
                config={'displayModeBar': False}
            ),

        ], id="crazy"),
        # dbc.ModalFooter(dbc.Button("Close", id="save-actions", class_name="me-1"))
    ], id="modal", is_open=False, size="xl", zIndex=99999, autoFocus=False, centered=True),




    ###################### Settings ##############################
    html.Button("Settings", id="settings-btn",
                style={"position":"absolute","border":"none","right":"0vw","top":"0vh"}),
    
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("1. Import Data")),
        dbc.ModalBody([
            
            dcc.Upload(id="upload-data",
                       children=html.Div([
                           "Drag and Drop or ",
                           html.A("Select File", style={"cursor":"pointer","color":"#800080","fontWeight":"600"})
                        ]),
                        multiple=False,
                        style={'height':'60px','lineHeight':'60px','borderWidth':'1px','borderStyle':'dashed',
                               'borderRadius':'5px','textAlign':'center','margin':'10px'},

            ),

            html.Div(id="output-data-upload"),

        ]),
        dbc.ModalFooter(dbc.Button("Next", id="get-data-next-btn", class_name="me-1"))
    ], id="get-data-modal", is_open=False, size="xl", zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),


    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("2. Select Output")),
        dbc.ModalBody([
            
            dcc.Dropdown(
                id="select-output-dropdown",
                multi=False,
                clearable=False,
                placeholder="Select Output...",
                optionHeight=40,
                disabled=False,
                style={"fontSize":"14px", "border":"none", "cursor":"pointer"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                # className="some-class",
            ),

        ]),
        dbc.ModalFooter(dbc.Button("Next", id="save-output-next-btn", class_name="me-1"))
    ], id="select-output-modal", is_open=False, size="lg", zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("3. Does The Order Of Samples Matter?")),
        dbc.ModalBody([
            
            dcc.Dropdown(
                id="sample-order-dropdown",
                multi=False,
                clearable=False,
                options=["No","Yes"],
                value="No",
                # placeholder="Select Output...",
                optionHeight=40,
                disabled=False,
                style={"fontSize":"14px", "border":"none", "cursor":"pointer"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                # className="some-class",
            ),

        ]),
        dbc.ModalFooter(dbc.Button("Next", id="sample-order-next-btn", class_name="me-1"))
    ], id="select-sample-order-modal", is_open=False, size="lg", zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),


    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("4. Train Model")),
        dbc.ModalBody([
            
            html.Div([
                html.Div("PARAMETERS", style={"fontWeight":"700"}),
                html.Div(style={"height":"10px"}),
                html.Div("Number of Estimators"),
                html.Div(style={"height":"5px"}),
                dcc.Slider(
                    min=1, max=500, step=1, marks=None,
                    value=200,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="n-estimators",
                    className="slider-class",#
                ),
                html.Div("Learning Rate"),
                html.Div(style={"height":"5px"}),
                dcc.Slider(
                    min=0.01, max=0.5, step=0.01, marks=None,
                    value=0.1,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="learning-rate",
                    className="slider-class",#
                ),
                html.Div("Tree Depth"),
                html.Div(style={"height":"5px"}),
                dcc.Slider(
                    min=1, max=10, step=1, marks=None,
                    value=2,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="max-depth",
                    className="slider-class",#
                ),
                html.Div("Colsample By Node"),
                html.Div(style={"height":"5px"}),
                dcc.Slider(
                    min=0.1, max=1.0, step=0.1, marks=None,
                    value=0.1,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="colsample-bynode",
                    className="slider-class",#
                ),
                html.Div(style={"height":"50px"}),
                dbc.Button("Train", id="train-model", class_name="me-1", style={"width":"100%","height":"80px"}),

            ], style={"border":"1px solid","display":"inline-block","width":"200px","padding":"5px 10px 10px 10px","borderRadius":"10px"}),

            html.Div([
                html.Div("RESULTS", style={"fontWeight":"700"}),
                html.Div(style={"height":"10px"}),
                # html.Div("ACCURACY", style={"fontWeight":"700"}),
                # html.Div("68%", id="model-accuracy",
                #          style={"fontWeight":"600","fontSize":"50px","height":"200px","display":"flex","border":"1px solid",
                #                        "justifyContent":"center","alignItems":"center","textAlign":"center"}),
                html.Div([

                    dcc.Graph(
                        id="model-accuracy-gauge",
                        figure=get_indicator_KPI(99.912, 87.5234, "Accuracy %"),
                        # style={"height":"100%","width":"100%","padding":"0px", "border":"1px solid",
                        #     "boxSizing":"border-box"},
                        style={"border":"0px solid","boxSizing":"border-box","display":"inline-block"},
                        config={'displayModeBar': False}
                    ),

                    html.Div(style={"display":"inline-block","border":"0px solid","height":"180px","width":"200px","boxSizing":"border-box"}),


                ]),
                

                dcc.Graph(
                    id="cv-graph",
                    style={"height":"150px","width":"520px","padding":"5px", "border":"0px solid",
                        "boxSizing":"border-box"},
                    config={'displayModeBar': False}
                ),

            ], id="something",
            style={"display":"inline-block","width":"550px","border":"1px solid","verticalAlign":"top","marginLeft":"10px",
                   "height":"390px","padding":"5px 10px 10px 10px","borderRadius":"10px"}),

            # html.Div([
            #     dcc.Graph(
            #         id="cv-graph",
            #         style={"height":"100%","width":"100%","padding":"5px", 
            #             "boxSizing":"border-box"},
            #     ),
            # ], style={"display":"inline-block","width":"300px","border":"1px solid","verticalAlign":"top","marginLeft":"10px",
            #           "height":"310px","borderRadius":"10px"})
        
        ]),
        dbc.ModalFooter(dbc.Button("Done", id="train-model-close-btn", class_name="me-1"))
    ], id="train-model-modal", is_open=False, size="lg", zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),

    dcc.Store(id="previous-model-score", data=None),




], id="main-window2", style={"position":"relative","width":"100%","height":"100vh","backgroundColor":"#f1f1f1","border":"0px solid"})


def get_graph(x, y, x_title, y_title, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=10),#t=35
        title=dict(text=title, font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=16, color="black"),
        showlegend=False,
        xaxis = dict(title=dict(text=x_title)),
        yaxis = dict(title=dict(text=y_title)),
    )
    return fig

def get_multi_trace_graph(xs, ys, names, x_title, y_title, title, show_legend=True):
    fig = go.Figure()
    for i in range(0, len(ys)):
        fig.add_trace(go.Scatter(
            x=xs[i],
            y=ys[i],
            name=names[i],
            mode='lines+markers',
        ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=0),#t=35
        title=dict(text=title, font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=14, color="black"),
        showlegend=show_legend,
        # xaxis=dict(title=dict(text=x_title)),
        yaxis=dict(title=dict(text=y_title, font=dict(size=14)), range=[0,1]),
    )
    return fig





############################## Closes the Train Model UI
@app.callback(
    dash.dependencies.Output(component_id="train-model-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Input(component_id="train-model-close-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def close_train_model_modal(n_clicks):
    return False


############################## Handles the Train Model UI
@app.callback(
    dash.dependencies.Output(component_id="model-accuracy-gauge", component_property='figure'),
    dash.dependencies.Output(component_id="cv-graph", component_property='figure'),
    dash.dependencies.Output(component_id="update-page-btn", component_property='n_clicks'),
    # dash.dependencies.Output(component_id="page-title", component_property='children'),
    dash.dependencies.Output(component_id="previous-model-score", component_property='data'),
    
    dash.dependencies.Input(component_id="train-model", component_property='n_clicks'),
    dash.dependencies.State(component_id="n-estimators", component_property='value'),
    dash.dependencies.State(component_id="learning-rate", component_property='value'),
    dash.dependencies.State(component_id="max-depth", component_property='value'),
    dash.dependencies.State(component_id="colsample-bynode", component_property='value'),
    dash.dependencies.State(component_id="update-page-btn", component_property='n_clicks'),
    dash.dependencies.State(component_id="previous-model-score", component_property='data'),
    prevent_initial_call=True,
)
def train_model(n_clicks, n_estimators, learning_rate, max_depth, colsample_bynode, update_page_n_clicks, prev_model_score):
    print("in train_model")
    global model

    if order_matters:
        cv = TimeSeriesSplit(n_splits=5)
    else:
        cv = 5

    if is_classification:
        print("classification")
        df_clean_target = df.dropna(subset=target)
        y = np.where(df_clean_target[target] == target_map[0], 0, 1)
        X = df_clean_target.drop(columns=target)
        encoded_X = pd.get_dummies(X, prefix_sep="?")
        print(encoded_X.shape)
        model = xgb.XGBClassifier(n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  colsample_bynode=colsample_bynode, 
                                  gamma=1,
                                  random_state=0)
        # scores = cross_val_score(model, encoded_X, y, cv=5, scoring="roc_auc")# 0.6208767361111109
        results = cross_validate(model, encoded_X, y, cv=cv, scoring="roc_auc", return_train_score=True)# 0.6208767361111109
        print(results["test_score"])
        # print(scores)
        model.fit(encoded_X, y)
    else:
        print("regression")
        df_clean_target = df.dropna(subset=target)
        y = df_clean_target[target]
        X = df_clean_target.drop(columns=target)
        encoded_X = pd.get_dummies(X, prefix_sep="?")
        print(encoded_X.shape)
        model = xgb.XGBRegressor(n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                colsample_bynode=colsample_bynode,
                                gamma=1,
                                random_state=0)
        # scores = cross_val_score(model, encoded_X, y, cv=5, scoring="r2")# 0.6208767361111109
        results = cross_validate(model, encoded_X, y, cv=cv, scoring="r2", return_train_score=True)# 0.6208767361111109
        print(results["test_score"])
        # print(scores)
        model.fit(encoded_X, y)
    
    update_model_metadata(model, df_clean_target)

    folds = [f"Fold {i+1}" for i in range(0, 5)]
    train_scores = np.maximum(results["train_score"], 0)
    test_scores = np.maximum(results["test_score"], 0)
    curr_model_score = test_scores.mean()*100
    if prev_model_score == None:
        prev_model_score = curr_model_score
    model_accuracy_gauge = get_indicator_KPI(curr_value=curr_model_score, prev_value=prev_model_score, title="Accuracy %")
    cv_graph = get_multi_trace_graph(xs=[folds, folds], ys=[train_scores,test_scores], names=["Train","Test"], x_title="", y_title="Accuracy", title="")
    
    # f"{results['test_score'].mean()*100:.2f}%"
    return model_accuracy_gauge, cv_graph, update_page_n_clicks + 1, curr_model_score


############################## Opens the Train Model UI
@app.callback(
    dash.dependencies.Output(component_id="train-model-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="select-sample-order-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Input(component_id="sample-order-next-btn", component_property='n_clicks'),
    dash.dependencies.State(component_id="sample-order-dropdown", component_property='value'),
    prevent_initial_call=True,
)
def open_train_model_modal(n_clicks, sample_order_matters):
    print("inside open_train_model_modal")
    global order_matters

    if sample_order_matters == "No":
        order_matters = False
    else:
        order_matters = True
    return True, False


# ############################## Handles the Sample Order UI
# @app.callback(
#     dash.dependencies.Output(component_id="select-sample-order-modal", component_property='is_open', allow_duplicate=True),
#     dash.dependencies.Input(component_id="sample-order-dropdown", component_property='value'),
#     prevent_initial_call=True,
# )
# def update_sample_order_variable(sample_order_matters):
#     print("inside update_sample_order_variable")
#     global order_matters

#     if sample_order_matters == "No":
#         order_matters = False
#     else:
#         order_matters = True
#     return dash.no_update



############################## Opens the Sample Order UI
@app.callback(
    dash.dependencies.Output(component_id="select-sample-order-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="select-output-modal", component_property='is_open'),
    dash.dependencies.Input(component_id="save-output-next-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def open_sample_order_modal(n_clicks):
    return True, False



############################## Handles the Select Output UI
@app.callback(
    dash.dependencies.Output(component_id="select-output-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Input(component_id="select-output-dropdown", component_property='value'),
    prevent_initial_call=True,
)
def update_output_variable(target_):
    print("inside update_output_variable")
    global target
    global is_classification
    global target_map

    if target_ is None:
        return dash.no_update
    
    print(target_)
    print(df.shape)
    target = target_
    unique_vals = np.unique(df[target_].dropna())
    if len(unique_vals) == 2:
        is_classification = True
        target_map = {0:unique_vals[0], 1:unique_vals[1]}
    elif (len(unique_vals) > 2) and is_numeric_dtype(df[target_]):
        is_classification = False
    return dash.no_update


############################## Opens the Select Output UI
@app.callback(
    dash.dependencies.Output(component_id="select-output-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="get-data-modal", component_property='is_open'),
    dash.dependencies.Input(component_id="get-data-next-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def open_select_output_modal(n_clicks):
    return True, False


############################## Handles the Import Data UI
@app.callback(
    dash.dependencies.Output(component_id="output-data-upload", component_property='children'),
    dash.dependencies.Output(component_id="select-output-dropdown", component_property='options'),
    dash.dependencies.Input(component_id="upload-data", component_property='contents'),
    dash.dependencies.Input(component_id="upload-data", component_property='filename'),
    dash.dependencies.Input(component_id="upload-data", component_property='last_modified'),
    prevent_initial_call=True,
)
def show_uploaded_data(contents, filename, modified_date):
    global df
    print("inside show_uploaded_data")
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif "xls" in filename:
            df = pd.read_excel(io.BytesIO(decoded), sheet_name=0, engine="openpyxl")
        n_row, n_col = df.shape
        df_head = df.head(10)
        children = html.Div([
            html.H5(filename),
            html.Div([
                html.H6("Data Preview...", style={"display":"inline-block"}),
                # formatted_number = f"{number:,}"
                html.H6(f"Rows = {n_row:,} | Columns = {n_col:,}", style={"display":"inline-block","float":"right"}),
            ]),
            # html.H6(f"Rows = {n_row}, Cols = {n_col}    Data Preview..."),
            dag.AgGrid(
                rowData=df_head.to_dict('records'),
                columnDefs=[{'field': i} for i in df.columns],
                defaultColDef = {"resizable": True},
                columnSize="autoSize",#"sizeToFit"
            ),
        ])
        return children, df.columns


############################## Opens the Import Data UI
@app.callback(
    dash.dependencies.Output(component_id="get-data-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Input(component_id="settings-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def open_get_data_modal(n_clicks):
    print("inside open_settings")
    return True



@app.callback(
    dash.dependencies.Output(component_id="feature-slider-container", component_property='children'),
    dash.dependencies.Output(component_id="positive-contribution-container", component_property='children'),
    dash.dependencies.Output(component_id="negative-contribution-container", component_property='children'),
    dash.dependencies.Output(component_id="page-title", component_property='children'),
    dash.dependencies.Input(component_id="update-page-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def clicked_button(n_clicks):
    print("inside clicked_button")
    df_clean_target = df.dropna(subset=target)
    feature_sliders = []
    positive_contributions_containers = []
    negative_contributions_containers = []
    # for i in range(0, len(df_clean_target.columns)):
    for i in range(0, len(ordered_features)):
        # feature = df_clean_target.columns[i]
        feature = ordered_features[i]
        if df_clean_target[feature].dtype == object:
            categories = np.unique(df_clean_target[feature].dropna())
            feature_slider = html.Div([
                html.Div(feature, title=feature, id={"type":"click-features", "index":f"{i}-{feature}"}, n_clicks=0,
                         style={"lineHeight":"1.5","marginBottom":"0px","marginTop":"2px","whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis","cursor":"pointer"}),
                dcc.Dropdown(
                    id={"type":"update-x-values", "index":f"{i}-{feature}"},
                    options=categories,
                    value=df_clean_target[feature].mode().iloc[0],
                    multi=False,
                    clearable=False,
                    placeholder="",
                    optionHeight=40,
                    disabled=False,
                    style={"fontSize":"12px", "border":"none"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                )
            ], style={"width":"15vw","border":"1px solid #cccccc","fontSize":"14px","height":"60px","paddingLeft":"10px","borderRadius":"5px","backgroundColor":"#ffffff"})
            feature_sliders.append(feature_slider)



            # contrib = 10
            positive_container = html.Div([
                html.Div(
                    # style={"width":f"{contrib}vw","height":"40px", "marginTop":"10px", "border":"1px solid","borderLeft":"none"}),
                    style={"position":"absolute","width":"0vw","height":"40px", "top":"10px", "border":"0px solid","borderLeft":"none","backgroundColor":"#599ad3"}),
                html.Div("",
                         style={"position":"absolute","top":"20px","left":"0vw"})
            ], id={"type":"positive-contributions-div", "index":f"{i}-{feature}"},
            style={"position":"relative", "width":"15vw","border":"1px solid transparent","borderLeft":"none","fontSize":"14px",
                   "height":"60px","fontWeight":"600"})
            positive_contributions_containers.append(positive_container)


            # contrib = 10
            negative_container = html.Div([
                html.Div(
                    # style={"width":f"{contrib}vw","height":"40px", "marginTop":"10px", "float":"right", "border":"1px solid","borderRight":"none"}),
                    style={"position":"absolute","width":"0vw","height":"40px", "top":"10px", "right":"0px", "border":"0px solid","borderRight":"none","backgroundColor":"#f9a65a"}),
                html.Div("",
                         style={"position":"absolute","top":"20px","right":"0vw"})
            ], id={"type":"negative-contributions-div", "index":f"{i}-{feature}"},
            style={"position":"relative", "width":"15vw","border":"1px solid transparent","borderRight":"none","fontSize":"14px",
                   "height":"60px","fontWeight":"600"})
            negative_contributions_containers.append(negative_container)
        else:
            feature_min = df_clean_target[feature].quantile(0.05)
            feature_max = df_clean_target[feature].quantile(0.95)
            # feature_min = df[feature].min()
            # feature_max = df[feature].max()
            precision_size = (feature_max - feature_min) / 50
            if precision_size >= 1:
                rounded_places = 0
                feature_step = 1
            elif precision_size >= 0.1:
                rounded_places = 1
                feature_step = 0.1
            elif precision_size >= 0.01:
                rounded_places = 2
                feature_step = 0.01
            else:
                rounded_places = 3
                feature_step = 0.001
            feature_min = feature_min.round(rounded_places)
            feature_max = feature_max.round(rounded_places)


            feature_slider = html.Div([
                html.Div(feature, title=feature, id={"type":"click-features", "index":f"{i}-{feature}"}, n_clicks=0,
                         style={"lineHeight":"1.5","marginBottom":"11px","marginTop":"2px","whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis","cursor":"pointer"}),
                dcc.Slider(
                    min=feature_min, max=feature_max, step=feature_step, marks=None,
                    value=df_clean_target[feature].median().round(rounded_places),
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"13px"}},
                    id={"type":"update-x-values", "index":f"{i}-{feature}"},
                    className="slider-class",#
                )
            ], style={"width":"15vw","border":"1px solid #cccccc","fontSize":"14px","height":"60px","paddingLeft":"10px","borderRadius":"5px","backgroundColor":"#ffffff"})
            feature_sliders.append(feature_slider)



            # contrib = 10
            positive_container = html.Div([
                html.Div(
                    # style={"width":f"{contrib}vw","height":"40px", "marginTop":"10px", "border":"1px solid","borderLeft":"none"}),
                    style={"position":"absolute","width":"0vw","height":"40px", "top":"10px", "border":"0px solid","borderLeft":"none","backgroundColor":"#599ad3"}),
                html.Div("",
                         style={"position":"absolute","top":"20px","left":"0vw"})
            ], id={"type":"positive-contributions-div", "index":f"{i}-{feature}"},
            style={"position":"relative", "width":"15vw","border":"1px solid transparent","borderLeft":"none","fontSize":"14px",
                   "height":"60px","fontWeight":"600"})
            positive_contributions_containers.append(positive_container)


            # contrib = 10
            negative_container = html.Div([
                html.Div(
                    # style={"width":f"{contrib}vw","height":"40px", "marginTop":"10px", "float":"right", "border":"1px solid","borderRight":"none"}),
                    style={"position":"absolute","width":"0vw","height":"40px", "top":"10px", "right":"0px", "border":"0px solid","borderRight":"none","backgroundColor":"#f9a65a"}),
                html.Div("",
                         style={"position":"absolute","top":"20px","right":"0vw"})
            ], id={"type":"negative-contributions-div", "index":f"{i}-{feature}"},
            style={"position":"relative", "width":"15vw","border":"1px solid transparent","borderRight":"none","fontSize":"14px",
                   "height":"60px","fontWeight":"600"})
            negative_contributions_containers.append(negative_container)

    if is_classification:
        if target_map[1] == 1:
            page_title = f"{target.upper()} - SIMULATION"
        else:
            page_title = f"{target_map[1].upper()} - SIMULATION"
    else:
        page_title = f"{target.upper()} - SIMULATION"

    return feature_sliders, positive_contributions_containers, negative_contributions_containers, page_title




@app.callback(
    dash.dependencies.Output(component_id={"type":"positive-contributions-div", "index":ALL}, component_property='children'),
    dash.dependencies.Output(component_id={"type":"negative-contributions-div", "index":ALL}, component_property='children'),
    dash.dependencies.Output(component_id="model-prediction", component_property='children'),
    dash.dependencies.Output(component_id="modal", component_property='is_open'),
    dash.dependencies.Output(component_id="contrib-graph", component_property='figure'),
    dash.dependencies.Input(component_id={"type":"update-x-values", "index":ALL}, component_property='value'),
    dash.dependencies.Input(component_id={"type":"click-features", "index":ALL}, component_property='n_clicks'),
    dash.dependencies.State(component_id={"type":"positive-contributions-div", "index":ALL}, component_property='children'),
    dash.dependencies.State(component_id={"type":"negative-contributions-div", "index":ALL}, component_property='children'),
    prevent_initial_call=True,
)
def update_prediction(ordered_values, n_clicks, positive_contributions_div, negative_contributions_div):
    print("inside update_prediction")
    # print(ctx.triggered_id["type"])
    # print(n_clicks)


    # Get update feature and index
    indx_feature = ctx.triggered_id["index"].split("-")
    indx = int(indx_feature[0])
    selected_feature = indx_feature[1]
    print(selected_feature)
    value = ctx.triggered[0]["value"]


    # Get all features...............Consider setting an id to the feature div, and use it's children as a state. No need to loop
    input_info = ctx.inputs_list[0]
    ordered_features = []
    ordered_features_with_category = []
    for i in range(0, len(input_info)):
        feature = input_info[i]["id"]["index"].split("-")[1]
        if feature in categorical_features:
            ordered_features_with_category.append(feature + "?" + ordered_values[i])
            ordered_values[i] = 1
        else:
            ordered_features_with_category.append(feature)
        ordered_features.append(feature)
    # print(ordered_features)

    # Form ordered_X dataframe
    ordered_X = pd.DataFrame(np.reshape(ordered_values, (1,-1)), columns=ordered_features_with_category)
    # print(ordered_X)

    # Get model ready X
    model_ready_X = ordered_X.reindex(columns=model_features, fill_value=0)
    # print(model_ready_X)


    if (ctx.triggered_id["type"] == "click-features") and (np.any(n_clicks)):
        print("doing click feature stuff")
        if selected_feature in categorical_features:
            feature_values = categorical_features[selected_feature]
            model_ready_X = model_ready_X.loc[model_ready_X.index.repeat(len(feature_values))].reset_index(drop=True)
            cat_feature_cols = [selected_feature + "?" + cat for cat in feature_values]
            model_ready_X[cat_feature_cols] = np.identity(len(cat_feature_cols))
        else:
            feature_values = np.unique(df[selected_feature])
            if len(feature_values) > 1000:
                feature_values = np.linspace(start=feature_values[0], stop=feature_values[-1], num=1000)
            model_ready_X = model_ready_X.loc[model_ready_X.index.repeat(len(feature_values))].reset_index(drop=True)
            model_ready_X[selected_feature] = feature_values

        if is_classification:
            predicted_vals = model.predict_proba(model_ready_X)[:,1]
        else:
            predicted_vals = model.predict(model_ready_X)
        contrib_fig = get_graph(x=feature_values, y=predicted_vals, x_title=selected_feature, y_title="Prediction", title=f"{selected_feature} vs Prediction")
        
        # feature_index = np.where(model_ready_X.columns == selected_feature)[0][0]
        # contribs = model.get_booster().predict(xgb.DMatrix(model_ready_X), pred_contribs=True)
        # contrib_fig = get_graph(x=feature_values, y=contribs[:,feature_index], x_title=selected_feature, y_title="Contribution", title=f"{selected_feature} Contributions")
        return [dash.no_update]*len(positive_contributions_div), [dash.no_update]*len(negative_contributions_div), dash.no_update, True, contrib_fig
        

    # Get prediction contributions
    contribs = model.get_booster().predict(xgb.DMatrix(model_ready_X), pred_contribs=True)
    bias = contribs[0, -1]
    contribs = contribs[:, :-1]
    
    # Store prediction contributions in a dataframe
    contrib_df = pd.DataFrame(contribs, columns=model_ready_X.columns)
    # print(contrib_df)

    # Aggregate SHAP values for categorical features
    contrib_df = contrib_df.groupby(lambda x: x.split('?')[0], axis=1).sum()
    
    # Reorder the dataframe columns to match the web app and store into array
    ordered_contribs = np.asarray(contrib_df[ordered_features])[0]


    # Update contribution bars and labels
    max_width = 15# max width of container
    for i in range(0, len(ordered_contribs)):
        if ordered_contribs[i] >= 0:
            new_contrib = ordered_contribs[i]
            new_width = (new_contrib / max_contrib) * max_width

            positive_contributions_div[i][0]["props"]["style"]["width"] = f"{new_width}vw"
            positive_contributions_div[i][1]["props"]["children"] = f"+{new_contrib:.{sig_figs}f}"
            positive_contributions_div[i][1]["props"]["style"]["left"] = f"{new_width}vw"

            negative_contributions_div[i][0]["props"]["style"]["width"] = "0vw"
            negative_contributions_div[i][1]["props"]["children"] = ""
            negative_contributions_div[i][1]["props"]["style"]["right"] = "0"
        else:
            new_contrib = -ordered_contribs[i]
            new_width = (new_contrib / max_contrib) * max_width

            positive_contributions_div[i][0]["props"]["style"]["width"] = "0vw"
            positive_contributions_div[i][1]["props"]["children"] = ""
            positive_contributions_div[i][1]["props"]["style"]["left"] = "0"

            negative_contributions_div[i][0]["props"]["style"]["width"] = f"{new_width}vw"
            negative_contributions_div[i][1]["props"]["children"] = f"-{new_contrib:.{sig_figs}f}"
            negative_contributions_div[i][1]["props"]["style"]["right"] = f"{new_width}vw"


    # Get prediction
    if is_classification:
        tot_contribs = ordered_contribs.sum() + bias
        prediction = (1 / (1 + np.exp(-tot_contribs)))*100
        if target_map[1] == 1:
            class1_display = f"{target} (prob)"
        else:
            class1_display = f"{target_map[1]} (prob)"
        prediction_div = [
            html.Div([
                html.Div([
                    html.Span(f"{prediction:.{1}f}", style={"border":"0px solid","lineHeight":"1"}, title=f"{prediction}%"), 
                    html.Span("%",style={"border":"0px solid","fontSize":"2.5vh","marginBottom":"1vh"}),
                ], style={"width":"100%","border":"0px solid","marginTop":"1vh"}),
                
                html.Div(class1_display, style={"border":"0px solid","fontSize":"2vh","lineHeight":"1","paddingBottom":"1vh"}),
            ], style={"position":"relative","width":"100%","border":"0px solid","textAlign":"center","color":"#333333"})
        ]
    else:
        prediction = ordered_contribs.sum() + bias
        prediction_div = [
            html.Div([
                html.Div([
                    html.Span(f"{prediction:.{sig_figs}f}", style={"border":"0px solid","lineHeight":"1"}, title=str(prediction)), 
                ], style={"width":"100%","border":"0px solid","marginTop":"1vh"}),
                
                html.Div(target, style={"border":"0px solid","fontSize":"2vh","lineHeight":"1","paddingBottom":"1vh"}),
            ], style={"position":"relative","width":"100%","border":"0px solid","textAlign":"center","color":"#333333"}),
        ]

    return positive_contributions_div, negative_contributions_div, prediction_div, False, dash.no_update
    # return positive_contributions_div, negative_contributions_div, prediction_div, str(prediction), False, dash.no_update
    # return positive_contributions_div, negative_contributions_div, f"{prediction:.{sig_figs}f}", str(prediction), False, dash.no_update




# @app.callback(
#     # dash.dependencies.Output(component_id="model-prediction", component_property='children'),
#     dash.dependencies.Output(component_id={"type":"positive-contributions-label", "index":ALL}, component_property='children'),
#     dash.dependencies.Input(component_id={"type":"update-x-values", "index":ALL}, component_property='value'),
#     dash.dependencies.State(component_id={"type":"positive-contributions-label", "index":ALL}, component_property='children'),
#     prevent_initial_call=True,
# )
# def update_prediction(slider_values, labels):
#     print("inside update_prediction")
#     indx_feature = ctx.triggered_id["index"].split("-")
#     indx = int(indx_feature[0])
#     feature = indx_feature[1]
#     print(feature)
#     # print(ctx.triggered_id)
#     value = ctx.triggered[0]["value"]

#     # p = Patch()
#     # p[feature] = value
#     print(labels[2])
#     labels[indx] = value
#     # print()
#     # print(value)
#     return labels


    



# app.layout = html.Div([

#     html.Div(style={"height":"10vh"}),

#     html.Div([

#         html.Div(
#             html.Div("LOAD CELL 12 ACTUAL TENSION iN THE WHOLE OF", 
#                      style={"width":"100%","height":"100%","border":"1px solid", "display":"flex", "justifyContent":"center", 
#                             "alignItems":"center","lineHeight":"1"})

#         , style={"width":"15vw","height":"5vh","border":"0px solid","display":"inline-block","verticalAlign":"middle"}),

#         html.Div(
#             dcc.Slider(
#                 min=0, max=5, step=1, marks=None,
#                 value=3,
#                 tooltip={"placement": "right", "always_visible": True},# updatemode="drag", updatemode="mouseup"
#                 id="my-slider", className="someclass"
#             )
#         , style={"width":"8vw","border":"0px solid","display":"inline-block","verticalAlign":"middle"}),

#     ], style={"fontSize":"1.8vh","border":"1px solid"}),


#     html.Div([

#         html.Div(
#             html.Div("LOAD CELL 12 ACTUAL TENSION iN THE WHOLE OF", 
#                      style={"width":"100%","height":"100%","border":"1px solid", "display":"flex", "justifyContent":"center", 
#                             "alignItems":"center","lineHeight":"1"})
        
#         , style={"width":"15vw","height":"5vh","border":"0px solid","display":"inline-block","verticalAlign":"middle"}),

#         html.Div(
#             dcc.Slider(
#                 min=0, max=5, step=1, marks=None,
#                 value=3,
#                 tooltip={"placement": "right", "always_visible": True},# updatemode="drag", updatemode="mouseup"
#                 id="my-slider", className="someclass"
#             )
#         , style={"width":"8vw","border":"0px solid","display":"inline-block","verticalAlign":"middle"}),

#     ], style={"fontSize":"1.8vh","border":"1px solid"})
    
# ], id="main-window2", style={"position":"relative","width":"100%","height":"100%"})


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)