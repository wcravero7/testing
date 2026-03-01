import dash
from dash import dcc, html, ctx
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import MATCH, ALL
from dash import dash_table
import plotly.graph_objects as go
import pickle
from sklearn.tree import DecisionTreeClassifier
# from sklearn.dummy import DummyClassifier
import math
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
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
# from flask import request


# app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, "style.css"])
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

def get_tags():
    tags_df = pd.read_csv("Tags.csv")
    # numerical_tags_series = pd.Series(tags_df.loc[tags_df["Agg Method"] == "mean", "Tags"], dtype=object)
    # numerical_tags_array = np.concatenate([numerical_tags_series + "__mean", numerical_tags_series + "__std"])
    # numerical_tags_array = np.insert(numerical_tags_array, 0, ["Shift","Product","Month"])# NEWWWW
    # numerical_tags_series = pd.Series(numerical_tags_array)
    # return numerical_tags_array, numerical_tags_series

    numerical_tags_series = pd.Series(tags_df.loc[tags_df["Agg Method"] == "mean", "Tags"], dtype=object)
    numerical_tags_array = np.concatenate([numerical_tags_series + "__mean", numerical_tags_series + "__std"])
    selectable_tags_array = np.insert(numerical_tags_array, 0, ["Shift","Product","Month"])
    filterable_tags_array = np.insert(numerical_tags_array, 0, ["Shift","Product"])
    selectable_tags_series = pd.Series(selectable_tags_array)
    return selectable_tags_array, selectable_tags_series, filterable_tags_array



# ############# Read Data From file #######################################
# with open("jul_to_aug_df.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "My Run Status"
# df[target] = np.where(df[target] == "R", 1, 0)

# ############# Read Data From file #######################################
# with open("bob_data_WaitTime.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "is_good"

############# Read Data From file #######################################
with open("bob_data.pickle", "rb") as file:
    df = pickle.load(file)
print(df.shape)
target = "is_good"

# ############# Read Data From file #######################################
# with open("bob_data_random.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "is_good"

# ############# Read Data From file #######################################
# with open("cancer_df.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "is_cancer"
# df[target] = np.random.randint(low=0, high=2, size=df.shape[0])



tree_depth = 1
gapY = 28
rootNode_centerY = (96 - tree_depth * gapY)/2# using 96 instead of 100 to add some padding
# gapX = 6.25

node_width = 7
node_height = 8#10
node_feature_gap = 0

# feature_width = 3.6 * gapX
feature_width = 15
feature_height = 7

nodes = []
feature_divs = []
connecting_lines = []
p_values = []
feature_font_size = 2

node_leftTopX = 40
node_leftTopY = 20
node_index_id = f"N{node_leftTopX},{node_leftTopY}"
# feature_left = node_leftTopX - (feature_width - node_width) / 2
# feature_top = node_leftTopY + node_height

# connector_width = 20
# connector_height = 10
# connector_left = feature_left - (connector_width - feature_width) / 2
# connector_top = feature_top + feature_height/2

# left_child_node_left = connector_left - node_width/2
# left_child_node_top = connector_top + connector_height
# right_child_node_left = connector_left + connector_width - node_width/2
# right_child_node_top = connector_top + connector_height



# overall_y = np.where(df["My Run Status"] == "R", 1, 0)
# overall_y_mean = overall_y.mean()
root_N = df.shape[0]
root_class1_N = int(df[target].sum())
root_class0_N = root_N - root_class1_N
root_class1_prob = root_class1_N / root_N


# app.layout = html.Div([
#     html.Div([
#         html.Div(round(root_class1_prob,3), style={"fontSize":"4vh","border":"0px solid","lineHeight":"1"}),
#         html.Div(f"n={root_N}", style={"fontSize":"2vh","height":"3vh","lineHeight":"1.2","border":"0px solid"}),
        
#     ], style={"position":"absolute","left":f"{node_leftTopX}%","top":f"{node_leftTopY}vh","width":f"{node_width}%","height":f"{node_height}vh","paddingTop":"1vh",
#               "border":"0px solid","textAlign":"center","boxSizing":"border-box","backgroundColor":"white","color":"black",
#               "boxShadow":"0 2px 6px 0 gray","borderRadius":"10px","zIndex":"4","cursor":"pointer"},
#         id={"type":"nodes", "index":node_index_id}, title=f"Samples: {root_N}\nClass 0: {root_class0_N}\nClass 1: {root_class1_N}"),
    
#     dbc.Modal([
#         dbc.ModalHeader(dbc.ModalTitle("Mitigation Practice", id="modal-title")),
#         dbc.ModalBody(["hey"], id="crazy"),
#         dbc.ModalFooter(dbc.Button("Save", id="save-actions", class_name="me-1"))
#     ], id="modal", is_open=False, size="lg", zIndex=99999),

# ], id="main-window", style={"position":"relative","width":"100%","height":"100%"})


app.layout = html.Div([

    html.Div([

        html.Div([
            html.Div(round(root_class1_prob,3), style={"fontSize":"4vh","border":"0px solid","lineHeight":"1"}),
            html.Div(f"n={root_N}", style={"fontSize":"2vh","height":"3vh","lineHeight":"1.2","border":"0px solid"}),
            
        ], style={"position":"absolute","left":f"{node_leftTopX}%","top":f"{node_leftTopY}vh","width":f"{node_width}%","height":f"{node_height}vh","paddingTop":"1vh",
                "border":"0px solid","textAlign":"center","boxSizing":"border-box","backgroundColor":"white","color":"black",
                "boxShadow":"0 2px 6px 0 gray","borderRadius":"10px","zIndex":"4","cursor":"pointer"},
            id={"type":"nodes", "index":node_index_id}, title=f"Samples: {root_N}\nClass 0: {root_class0_N}\nClass 1: {root_class1_N}"),

    ], id="main-window", style={"position":"relative","width":"100%","height":"100%"}),
    
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Mitigation Practice", id="modal-title")),
        dbc.ModalBody([
            
            # dcc.Graph(
            #     id="boxplot2",
            #     style={"position":"relative","height":"50vh","left":"0%","width":"30vw","top":"9vh","border":f"1px solid","padding":"5px", 
            #         "boxSizing":"border-box"},
            # ),

            # dcc.Graph(
            #     id="boxplot",
            #     style={"position":"relative","height":"50vh","left":"45vw","width":"30vw","top":"9vh","border":f"1px solid","padding":"5px", 
            #         "boxSizing":"border-box"},
            # ),


            html.Div([

                html.Div([
                    dcc.Graph(
                        id="boxplot2",
                        style={"position":"relative","height":"50vh","left":"0%","width":"100%","top":"9vh","border":f"1px solid","padding":"5px", 
                            "boxSizing":"border-box"},
                    ),
                ], style={"display":"inline-block","width":"60%"}),
                
                html.Div([

                    dcc.Graph(
                        id="boxplot",
                        style={"position":"relative","height":"50vh","left":"0%","width":"100%","top":"9vh","border":f"1px solid","padding":"5px", 
                            "boxSizing":"border-box","display":"inline-block"},
                    ),
                ], style={"display":"inline-block","width":"38%","marginLeft":"1%"})
                

            ], style={"border":"1px solid"})



        ], id="crazy"),
        dbc.ModalFooter(dbc.Button("Save", id="save-actions", class_name="me-1"))
    ], id="modal", is_open=False, size="xl", zIndex=99999, autoFocus=False, centered=True),#style={"position":"absolute"}
    # size="lg",fullscreen=True, 
    dcc.Store(id="feature-data"),

], id="main-window2", style={"position":"relative","width":"100%","height":"100%"})




def filter_df(str_splits):
    filter_mask = np.ones(df.shape[0], dtype=bool)
    for i in range(0, len(str_splits)):
        if "<" in str_splits[i]:
            temp = str_splits[i].split("<")
            col_index = int(temp[0])
            threshold = float(temp[1])
            filter_mask = filter_mask & (df.iloc[:, col_index] <= threshold)
        elif ">" in str_splits[i]:
            temp = str_splits[i].split(">")
            col_index = int(temp[0])
            threshold = float(temp[1])
            filter_mask = filter_mask & (df.iloc[:, col_index] > threshold)
        elif "=" in str_splits[i]:
            temp = str_splits[i].split("=")
            col_index = int(temp[0])
            filter_mask = filter_mask & (df.iloc[:, col_index] == temp[1])
        else:
            temp = str_splits[i].split("!")
            col_index = int(temp[0])
            filter_mask = filter_mask & (df.iloc[:, col_index] != temp[1])
    return df.loc[filter_mask, :].copy()


def get_X_and_Y_pos(str_splits):
    y_pos = len(str_splits)
    path = []
    for i in range(0, len(str_splits)):
        if "<" in str_splits[i]:
            path.append("<")
        else:
            path.append(">")
    if path == ["<","<"]:
        x_pos = 0
    elif path == ["<",">"]:
        x_pos = 1
    elif path == [">","<"]:
        x_pos = 2
    elif path == [">",">"]:
        x_pos = 3
    return x_pos, y_pos



def find_optimal_split(filtered_df, feature_name, class_weight):
    filtered_df_no_na = filtered_df[[feature_name, target]].dropna()

    y = filtered_df_no_na[target]
    X = filtered_df_no_na.drop(columns=target)
    # model = DecisionTreeClassifier(random_state=0, class_weight=class_weight, max_depth=1)
    model = DecisionTreeClassifier(random_state=0, class_weight=class_weight, max_depth=1, min_samples_leaf=50)
    model.fit(X, y)
    if model.tree_.node_count == 1:
        return None, False
    threshold = model.tree_.threshold[0]
    return threshold, True


def find_optimal_categorical_split(filtered_df, feature_name, class_weight):
    filtered_df_no_na = filtered_df[[feature_name, target]].dropna()
    # y = np.where(filtered_df_no_na["My Run Status"] == "R", 1, 0)
    y = filtered_df_no_na[target]
    X = pd.get_dummies(filtered_df_no_na[feature_name])
    model = DecisionTreeClassifier(random_state=0, class_weight=class_weight, max_depth=1)
    model.fit(X, y)
    if model.tree_.node_count == 1:
        return None, False
    selected_value = X.columns[model.tree_.feature[0]]
    return selected_value, True



def get_feature_scores(filtered_df):
    G_pop = filtered_df.loc[filtered_df[target] == 0, :]
    R_pop = filtered_df.loc[filtered_df[target] == 1, :]
    
    features = filtered_df.columns[filtered_df.columns != target]
    p_values = []
    for colname in features:
        G_pop_colname = G_pop[colname]
        R_pop_colname = R_pop[colname]
        if filtered_df[colname].dtype == object:
            contigency = pd.crosstab(filtered_df[colname], filtered_df[target])
            tbl = Table(contigency)# fisher's exact test approximation used for many categories
            p_value = tbl.test_nominal_association().pvalue# 0.02

            # means = filtered_df.groupby(colname)[target].mean()
            # encoded = filtered_df[colname].map(means)
            # p_value = roc_auc_score(filtered_df[target], encoded)
        else:
            if (G_pop_colname.shape[0] > 0) and (R_pop_colname.shape[0] > 0):
                # statistic, p_value = ks_2samp(G_pop_colname, R_pop_colname)
                # statistic, p_value = ttest_ind(G_pop_colname, R_pop_colname, equal_var=False)
                statistic, p_value = mannwhitneyu(G_pop_colname, R_pop_colname, alternative="two-sided")
                # p_value = statistic / (G_pop_colname.shape[0] * R_pop_colname.shape[0])
            else:
                p_value = np.nan
        p_values.append(p_value)

    p_values = np.array(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    sorted_features = features[sorted_indices]
    return sorted_features, sorted_indices, sorted_p_values


def get_boxplot(df, feature, font_size=16):
    good_vals = df.loc[df[target] == 0, feature]
    bad_vals = df.loc[df[target] == 1, feature]
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=good_vals,
        name="Steady<br>State",#"Steady State"
        # fillcolor="lightgreen",
        marker_color="green",
    ))
    fig.add_trace(go.Box(
        y=bad_vals,
        name="Before<br>UPS",#"Right Before UPS"
        # marker_color="red"
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=10),#t=35
        # title=dict(text="Histogram (Good vs Bad)", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=font_size, color="black"),# size=16
        showlegend=False,
        # xaxis_title=feature,
        # yaxis_title=feature,
    )
    return fig



@app.callback(
    dash.dependencies.Output(component_id="modal", component_property='is_open'),
    dash.dependencies.Output(component_id="boxplot", component_property='figure'),
    dash.dependencies.Input(component_id="feature-data", component_property='data'),
    prevent_initial_call=True,
)
def show_modal(data):
    print("SHOW MODAL....................................................................")
    filtered_df = pd.DataFrame(data)
    print(filtered_df.shape)
    boxplot = get_boxplot(df=filtered_df, feature=filtered_df.columns[0])
    return True, boxplot



@app.callback(
    dash.dependencies.Output(component_id="main-window", component_property='children'),
    dash.dependencies.Output(component_id="feature-data", component_property='data'),
    # dash.dependencies.Output(component_id="modal", component_property='is_open'),

    dash.dependencies.Input(component_id={"type":"nodes", "index":ALL}, component_property='n_clicks'),
    
    dash.dependencies.Input(component_id={"type":"remove-splits", "index":ALL}, component_property='n_clicks'),

    dash.dependencies.Input(component_id={"type":"radio-buttons", "index":ALL}, component_property='value'),

    dash.dependencies.Input(component_id={"type":"feature-nodes", "index":ALL}, component_property='n_clicks'),

    dash.dependencies.Input(component_id={"type":"hide-dropdowns", "index":ALL}, component_property='n_clicks'),

    dash.dependencies.Input(component_id={"type":"modal-charts", "index":ALL}, component_property='n_clicks'),

    dash.dependencies.State(component_id={"type":"nodes", "index":ALL}, component_property='style'),
    dash.dependencies.State(component_id={"type":"nodes", "index":ALL}, component_property='id'),
    dash.dependencies.State(component_id="main-window", component_property='children'),
    prevent_initial_call=True,
)
# def click_node(n_clicks, n_clicks_remove_split, split_features, n_clicks_feature_nodes, hide_dropdown, node_style, feature_id, all_nodes):
def click_node(n_clicks, n_clicks_remove_split, split_features, n_clicks_feature_nodes, hide_dropdown, click_modal_btn, node_style, feature_id, all_nodes):
    print("CLICKED NODE....................................................................")


    if ctx.triggered_id["type"] == "feature-nodes":
        print("CLICKED A FEATURE NODE")
        feature_dropdown_index_id = ctx.triggered_id["index"].replace("F","D",1)
        print(feature_dropdown_index_id)
        for i in range(0, len(all_nodes)):
            if all_nodes[i]["props"]["id"]["index"] == feature_dropdown_index_id:
                if all_nodes[i]["props"]["className"] == "closed-panel":
                    all_nodes[i]["props"]["className"] = "opened-panel"
                    print("Class Name: opened-panel")
                else:
                    all_nodes[i]["props"]["className"] = "closed-panel"
                    print("Class Name: closed-panel")
                break
        return all_nodes, dash.no_update
    

    if ctx.triggered_id["type"] == "hide-dropdowns":
        print("HIDING DROPDOWN")
        feature_dropdown_index_id = ctx.triggered_id["index"].replace("H","D",1)
        print(feature_dropdown_index_id)
        for i in range(0, len(all_nodes)):
            if all_nodes[i]["props"]["id"]["index"] == feature_dropdown_index_id:
                # all_nodes[i]["props"]["style"]["visibility"] = "hidden"
                all_nodes[i]["props"]["className"] = "closed-panel"
                break
        print("Finished hiding dropdown!")
        return all_nodes, dash.no_update



    if ctx.triggered_id["type"] == "remove-splits":
        print("IN REMOVE SPLITS!!!!!!!!!!!!!!!!!############################")
        new_nodes = []
        parent_index = ctx.triggered_id["index"].replace("E","N",1)
        print(parent_index)
        parent_splits = parent_index.split("|")[1:]
        parent_num_splits = len(parent_splits)
        print(parent_num_splits)

        for i in range(0, len(all_nodes)):
            other_index = all_nodes[i]["props"]["id"]["index"]
            other_splits = other_index.split("|")[1:1+parent_num_splits]
            if (other_index == parent_index) or (other_splits != parent_splits):
                new_nodes.append(all_nodes[i])
        return new_nodes, dash.no_update



    if ctx.triggered_id["type"] == "radio-buttons":
        new_nodes = []
        node_index_id = ctx.triggered_id["index"].replace("B","N",1)
        feature_dropdown_index_id = node_index_id.replace("N","D",1)
        print(node_index_id)
        node_splits = node_index_id.split("|")[1:]
        node_num_splits = len(node_splits)
        print(node_num_splits)

        for i in range(0, len(all_nodes)):
            other_index = all_nodes[i]["props"]["id"]["index"]
            other_splits = other_index.split("|")[1:1+node_num_splits]
            if (other_index == node_index_id) or (other_index == feature_dropdown_index_id) or (other_splits != node_splits):
                new_nodes.append(all_nodes[i])
                if (other_index == feature_dropdown_index_id):
                    # new_nodes[-1]["props"]["style"]["visibility"] = "hidden"
                    new_nodes[-1]["props"]["className"] = "closed-panel"
            
        print("CLICKED RADIO BUTTON-----------------------------------------------")
        all_nodes = new_nodes


    if ctx.triggered_id["type"] == "modal-charts":
        print("Clicked MODAL Btn")
        feature_node_index_id = ctx.triggered_id["index"].replace("M","F",1)
        print(feature_node_index_id)
        for i in range(0, len(all_nodes)):
            if all_nodes[i]["props"]["id"]["index"] == feature_node_index_id:
                print(all_nodes[i]["props"]["children"])
                feature_name = all_nodes[i]["props"]["children"]
                # node_index_arr = ctx.triggered_id["index"].split("|")
                node_index_arr = feature_node_index_id.split("|")
                filter_path = node_index_arr[1:]
                filtered_df = filter_df(filter_path)
                return all_nodes, filtered_df[[feature_name, target]].to_dict("list")



    if ctx.triggered_id["type"] == "nodes":
        # Get Clicked Node Index
        node_index_id = ctx.triggered_id["index"]
    

    # Get Clicked Node Index Array
    node_index_arr = node_index_id.split("|")


    # Get Filter Path Array
    # filter_path = node_index_arr[0:-1]
    filter_path = node_index_arr[1:]


    # Get Node Pos
    # pos_str = node_index_arr[-1][1:]
    pos_str = node_index_arr[0][1:]
    pos_list = pos_str.split(",")
    node_left = float(pos_list[0])
    node_top = float(pos_list[1])


    # Get class weights
    n_1 = df[target].sum()
    n_0 = df.shape[0] - n_1
    weight_0 = df.shape[0] / (2 * n_0)
    weight_1 = df.shape[0] / (2 * n_1)
    class_weight = {0:weight_0, 1:weight_1}


    # Filter df
    filtered_df = filter_df(filter_path)
    print(filtered_df.shape)
    

    # if ctx.triggered_id["type"] == "modal-charts":
    #     print("Clicked MODAL Btn")
    #     return all_nodes, filtered_df[["NDC Net1 Shape Magnitude", target]].to_dict("list")


    if ctx.triggered_id["type"] == "radio-buttons":
        selected_feature = ctx.triggered[0]["value"]
        selected_feature_index = np.where(filtered_df.columns == selected_feature)[0][0]
        

    if ctx.triggered_id["type"] == "nodes":
        sorted_features, sorted_indices, sorted_scores = get_feature_scores(filtered_df)
        selected_feature = sorted_features[0]
        selected_feature_index = np.where(filtered_df.columns == selected_feature)[0][0]
        print(selected_feature)

        # Create Radio Button Options
        feature_options = []
        for i in range(0, len(sorted_features)):
            feature_options.append(
                {
                    "label":[
                        html.Div(sorted_features[i], style={"width":"78%", "lineHeight":"1.0", "borderBottom":"0px solid"}),
                        html.Span(f"{sorted_scores[i]:.2f}", style={"marginLeft":"2%","borderBottom":"0px solid"})
                    ],
                    "value": sorted_features[i]
                },
            )
                


    if filtered_df[selected_feature].dtype == object:
        selected_value, is_splittable = find_optimal_categorical_split(filtered_df, selected_feature, class_weight)
        if not is_splittable:
            return dash.no_update, dash.no_update
        print("feature is categorical")
        left_child_threshold_label = f"={selected_value}"
        right_child_threshold_label = f"!={selected_value}"
        left_child_threshold_title = left_child_threshold_label
        right_child_threshold_title = right_child_threshold_label
        left_child_y = filtered_df.loc[filtered_df.iloc[:, selected_feature_index] == selected_value, target]
        right_child_y = filtered_df.loc[filtered_df.iloc[:, selected_feature_index] != selected_value, target]
        left_child_filter_str = f"{selected_feature_index}={selected_value}"
        right_child_filter_str = f"{selected_feature_index}!{selected_value}"
    else:
        selected_threshold, is_splittable = find_optimal_split(filtered_df, selected_feature, class_weight)
        if not is_splittable:
            return dash.no_update, dash.no_update
        print("featutre is numerical")
        print(selected_threshold)
        left_child_threshold_label = f"<{round(selected_threshold, 2)}"
        right_child_threshold_label = f">{round(selected_threshold, 2)}"
        left_child_threshold_title = f"<{selected_threshold}"
        right_child_threshold_title = f">{selected_threshold}"
        left_child_y = filtered_df.loc[filtered_df.iloc[:, selected_feature_index] <= selected_threshold, target]
        right_child_y = filtered_df.loc[filtered_df.iloc[:, selected_feature_index] > selected_threshold, target]
        left_child_filter_str = f"{selected_feature_index}<{selected_threshold}"
        right_child_filter_str = f"{selected_feature_index}>{selected_threshold}"




    # Left Child Metrics
    # left_child_y = filtered_df.loc[filtered_df.iloc[:, selected_feature_index] <= selected_threshold, target]
    left_child_UPS_mean = left_child_y.mean()
    left_child_UPS_N = left_child_y.shape[0]
    left_child_N = left_child_y.shape[0]
    left_child_class1_N = int(left_child_y.sum())
    left_child_class0_N = left_child_N - left_child_class1_N
    left_child_class1_prob = left_child_class1_N / left_child_N

    # Right Child Metrics
    # right_child_y = filtered_df.loc[filtered_df.iloc[:, selected_feature_index] > selected_threshold, target]
    right_child_UPS_mean = right_child_y.mean()
    right_child_UPS_N = right_child_y.shape[0]
    right_child_N = right_child_y.shape[0]
    right_child_class1_N = int(right_child_y.sum())
    right_child_class0_N = right_child_N - right_child_class1_N
    right_child_class1_prob = right_child_class1_N / right_child_N


    # Calculate significance of split
    _, split_p_value = fisher_exact([[left_child_class0_N, left_child_class1_N], [right_child_class0_N, right_child_class1_N]], alternative="two-sided")


    node_style = node_style[0]
    print(n_clicks)
    # print(node_style)
    node_width = 7
    node_height = 8#10

    feature_width = 20#15
    feature_height = 7
    feature_index_id = node_index_id.replace("N","F",1)
    feature_font_size = 2

    feature_left = node_left - (feature_width - node_width) / 2
    feature_top = node_top + node_height

    connector_width = 25#20
    connector_height = 10
    connector_left = feature_left - (connector_width - feature_width) / 2
    connector_top = feature_top + feature_height/2
    connector_index_id = node_index_id.replace("N","C",1)

    left_child_node_left = connector_left - node_width/2
    left_child_node_top = connector_top + connector_height
    right_child_node_left = connector_left + connector_width - node_width/2
    right_child_node_top = connector_top + connector_height

    threshold_label_width = connector_width/2# threshold_label_width = node_width
    threshold_label_height = 3
    left_child_threshold_label_left = left_child_node_left - (threshold_label_width - node_width)/2
    right_child_threshold_label_left = right_child_node_left - (threshold_label_width - node_width)/2


    if len(filter_path) == 0:
        left_child_node_index_id = f"N{left_child_node_left},{left_child_node_top}|{left_child_filter_str}"
        right_child_node_index_id = f"N{right_child_node_left},{right_child_node_top}|{right_child_filter_str}"
    else:
        left_child_node_index_id = f"N{left_child_node_left},{left_child_node_top}|" + "|".join(filter_path) + f"|{left_child_filter_str}"
        right_child_node_index_id = f"N{right_child_node_left},{right_child_node_top}|" + "|".join(filter_path) + f"|{right_child_filter_str}"

    
    left_child_threshold_index_id = node_index_id.replace("N","L",1)
    right_child_threshold_index_id = node_index_id.replace("N","R",1)

    feature_dropdown_index_id = node_index_id.replace("N","D",1)

    split_divs = [

    # Feature Label
    html.Div(selected_feature,
        style={"position":"absolute","left":f"{feature_left}%","top":f"{feature_top}vh","width":f"{feature_width}%",
                "height":f"{feature_height}vh", "border":"0px solid","fontSize":f"{feature_font_size}vh",
                "display":"flex","justifyContent":"center","alignItems":"center","textAlign":"center",
                "lineHeight":"1.2","backgroundColor":"white","boxShadow":"0 2px 6px 0 gray",
                "borderTopLeftRadius":"3px","borderTopRightRadius":"3px","borderBottomLeftRadius":"3px",
                "borderBottomRightRadius":"3px","zIndex":"5","cursor":"pointer"},
        id={"type":"feature-nodes", "index":feature_index_id}, title=f"p value: {split_p_value}"),


    # Connectors
    html.Div(style={"position":"absolute","left":f"{connector_left}%","top":f"{connector_top}vh","width":f"{connector_width}%",
                    "height":f"{connector_height}vh", "border":"1px solid", "borderBottom":"none", "zIndex":"1"},
             id={"type":"node-connectors", "index":connector_index_id}),

    # Left Child Node
    html.Div([
        html.Div(round(left_child_class1_prob,3), style={"fontSize":"4vh","border":"0px solid","lineHeight":"1"}),#left_child_UPS_mean
        html.Div(f"n={left_child_UPS_N}", style={"fontSize":"2vh","height":"3vh","lineHeight":"1.2","border":"0px solid"}),
        
    ], style={"position":"absolute","left":f"{left_child_node_left}%","top":f"{left_child_node_top}vh","width":f"{node_width}%","height":f"{node_height}vh","paddingTop":"1vh",
              "border":"0px solid","textAlign":"center","boxSizing":"border-box","backgroundColor":"white","color":"black",
              "boxShadow":"0 2px 6px 0 gray","borderRadius":"10px","zIndex":"3","cursor":"pointer"},
       id={"type":"nodes", "index":left_child_node_index_id}, title=f"Samples: {left_child_N}\nClass 0: {left_child_class0_N}\nClass 1: {left_child_class1_N}"),
       
    
    # Right Child Node
    html.Div([
        html.Div(round(right_child_class1_prob,3), style={"fontSize":"4vh","border":"0px solid","lineHeight":"1"}),#right_child_UPS_mean
        html.Div(f"n={right_child_UPS_N}", style={"fontSize":"2vh","height":"3vh","lineHeight":"1.2","border":"0px solid"}),
        
    ], style={"position":"absolute","left":f"{right_child_node_left}%","top":f"{right_child_node_top}vh","width":f"{node_width}%","height":f"{node_height}vh","paddingTop":"1vh",
              "border":"0px solid","textAlign":"center","boxSizing":"border-box","backgroundColor":"white","color":"black",
              "boxShadow":"0 2px 6px 0 gray","borderRadius":"10px","zIndex":"3","cursor":"pointer"},
       id={"type":"nodes", "index":right_child_node_index_id}, title=f"Samples: {right_child_N}\nClass 0: {right_child_class0_N}\nClass 1: {right_child_class1_N}"),

    
    # Left Threshold Label
    html.Div(left_child_threshold_label,
        style={"position":"absolute","left":f"{left_child_threshold_label_left}%","top":f"{left_child_node_top-threshold_label_height}vh","width":f"{threshold_label_width}%","height":f"{threshold_label_height}vh","paddingTop":"0vh",
               "border":"0px solid","textAlign":"center","boxSizing":"border-box","backgroundColor":"white","color":"black",
               "zIndex":"2","fontSize":"2vh"},
        id={"type":"threshold-labels", "index":left_child_threshold_index_id}, title=left_child_threshold_title),
    
    # Right Threshold Label
    html.Div(right_child_threshold_label,
        style={"position":"absolute","left":f"{right_child_threshold_label_left}%","top":f"{right_child_node_top-threshold_label_height}vh","width":f"{threshold_label_width}%","height":f"{threshold_label_height}vh","paddingTop":"0vh",
               "border":"0px solid","textAlign":"center","boxSizing":"border-box","backgroundColor":"white","color":"black",
               "zIndex":"2","fontSize":"2vh"},
        id={"type":"threshold-labels", "index":right_child_threshold_index_id}, title=right_child_threshold_title),


    ]


    if ctx.triggered_id["type"] == "nodes":
        split_divs.append(

            # Feature Dropdown
            html.Div([

                html.Div(style={"height":"3vh"}),
                html.Div([
                    html.Div("Tag", style={"width":"80%","display":"inline-block","marginLeft":"5%"}),
                    html.Div("Corr", style={"width":"15%", "display":"inline-block"}),
                ], style={"border":"0px solid","fontSize":"1.7vh"}),
                
                html.Div(style={"height":"0.5vh"}),
                dcc.RadioItems(
                    options=feature_options,
                    labelStyle={"display": "flex", "align-items": "center", "marginTop":"5px", "marginBottom":"5px"},
                    style={"border":"1px solid","marginLeft":"2%", "width":"96%","height":"25vh","overflowY":"scroll"},
                    id={"type":"radio-buttons", "index":node_index_id.replace("N","B",1)},
                    value=sorted_features[0],
                    className="hover-radio",
                ),

                html.Button("Additional Analytics", id={"type":"modal-charts", "index":node_index_id.replace("N","M",1)},
                            style={"marginLeft":"2%", "marginTop":"3vh", "borderRadius":"5px","border":"none","fontSize":"1.8vh", "fontWeight":500,
                                    "backgroundColor":"rgb(0, 136, 255)","color":"white","display":"inline-block","padding":"5px"}),

                html.Div([

                    html.Button("Remove Split", id={"type":"remove-splits", "index":node_index_id.replace("N","E",1)},
                                style={"marginLeft":"2%", "marginTop":"3vh", "borderRadius":"5px","border":"none","fontSize":"1.8vh","fontWeight":500,
                                       "backgroundColor":"rgb(0, 136, 255)","color":"white","display":"inline-block","padding":"5px"}),

                    html.Button("Cancel", id={"type":"hide-dropdowns", "index":node_index_id.replace("N","H",1)},
                                style={"float":"right", "marginRight":"2%", "marginTop":"3vh", "borderRadius":"5px","border":"none","fontSize":"1.8vh","fontWeight":500,
                                       "backgroundColor":"transparent","color":"rgb(0, 136, 255)","display":"inline-block"}),

                ]),

                html.Div(style={"height":"2vh"}),
                

            ], style={"position":"absolute","left":f"{feature_left}%","top":f"{feature_top + feature_height}vh","width":f"{feature_width}%",
                      "height":"auto", "border":"0px solid","fontSize":"1.5vh",#"visibility":"hidden",
                      "lineHeight":"1.2","backgroundColor":"white","boxShadow":"0 2px 6px 0 gray","borderBottomLeftRadius":"3px",
                      "borderBottomRightRadius":"3px","zIndex":"600"},
            id={"type":"feature-dropdowns", "index":feature_dropdown_index_id}, className="closed-panel"),

        )


    all_nodes.extend(split_divs)
    return all_nodes, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)