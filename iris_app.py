import dash
from dash import dcc
# import dash_core_components as dcc
from dash import html
# import dash_html_components as html
from dash import no_update, callback_context
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State
# import dash_table
from dash import dash_table
import networkx as nx
import plotly.graph_objs as go

from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json
import torch
import torchvision
import h5py
import numpy as np
import random
from PIL import Image
import os
import pandas as pd
import faulthandler
from search import *
import dash_bootstrap_components as dbc
import re
import subprocess

faulthandler.enable()

UPLOAD_DIRECTORY = "./data/images"
cyto.load_extra_layouts()

image_urls = []

def return_dimensions(bboxes):
    max_img_width = 0
    max_img_height = 0

    for box in bboxes:
        if box[0] > max_img_width:
            max_img_width = box[0]
        if box[2] > max_img_width:
            max_img_width = box[2]
        if box[1] > max_img_height:
            max_img_height = box[1]
        if box[3] > max_img_height:
            max_img_height = box[3]
    
    return max_img_width, max_img_height

def get_info_by_path(image_path, results, thres=0.5):
    project_dir = './data'
    vocab_file = json.load(open(f'{project_dir}/VG-SGG-dicts-with-attri.json'))
    
    prediction = results
    # boxes
    boxes = prediction['bbox']
    # predicted object labels
    idx2label = vocab_file['idx_to_label']
    pred_labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(prediction['extra_fields']['pred_labels'])]
    pred_scores = prediction['extra_fields']['pred_scores']
    # prediction relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    pred_rel_pair = prediction['extra_fields']['rel_pair_idxs']
    pred_rel_label = torch.FloatTensor(prediction['extra_fields']['pred_rel_scores'])
    # print(pred_rel_label)
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    mask = pred_rel_score > thres
    pred_rel_score = pred_rel_score[mask]
    pred_rel_label = pred_rel_label[mask]
    pred_rels = [(pred_labels[i[0]], idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]

    pred_rel_score = pred_rel_score.tolist()

    return image_path, boxes, pred_labels, pred_scores, pred_rels, pred_rel_score, pred_rel_label

def image_figure(image_path, w, h):
    # Create figure
    picture = Image.open(image_path)
    fig = go.Figure()

    # Constants
    img_width, img_height = w, h
    scale_factor = 0.6

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=picture)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig

def network_graph(image_path, predictions):
    image_path, boxes, pred_labels, pred_scores, pred_rels, pred_rel_score, pred_rel_label = get_info_by_path(image_path, predictions)

    default_stylesheet = [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'background-color': '#A6E22E',
                'width': '20',
                'height': '20',
            }
        },
        {
            'selector': 'edge',
            'style': {
                'line-color': '#FFFFFF',
                'width':1,
                'curve-style': 'bezier',
                'target-arrow-color': '#FFFFFF',
                'target-arrow-shape': 'triangle-backcurve',
                'target-arrow-fill': 'filled'
            }
        }
    ]

    nodes = []

    for index in range (0,len(pred_labels)):        
        nodes.append({'data': {'id': str(index), 'label': pred_labels[index]}})


    edges = []
    for relation in pred_rels:
        source_val = relation[0].split("-")[0]
        target_val = relation[2].split("-")[0]
        label_val = relation[0] + " " + relation[1] + " " + relation[2]
        if len(relation[1].split(" ")) == 1: 
            id_val = source_val + "-" + relation[1] + "-" + target_val
        else:
            id_val = source_val + "-" + relation[1].replace(" ", "_") + "-" + target_val
        edge = {'data': {'id': id_val, 'source': source_val, 'target': target_val, 'label': label_val}}
        edges.append(edge)
        
    cyto_graph = cyto.Cytoscape(
        id='cytoscape-scenegraph',
        layout={
            'name': 'spread',
            'fit': 'false',
            'minDist': '50',
        },
        elements=nodes+edges,
        stylesheet=default_stylesheet,
        style={'width': '100%', 'height': '100%'},
    )
        
    
    return cyto_graph, boxes, pred_rels, pred_rel_score

def load_dropdown_options(pred_rels):
    rel_dict = {}
    for relation in pred_rels:
        if relation[1] not in rel_dict.keys():
            rel_dict[relation[1]] = 0
    
    unique_rels =  list(rel_dict.keys())

    dropdown_options = [
        {'label': 'None', 'value': 'NIL'}
    ]

    for rel in unique_rels:
        dropdown_options.append({'label': rel, 'value': rel})
    
    return dropdown_options

def draw_scenegraph(jsonified_metadata):
    app_metadata = json.loads(jsonified_metadata)
    image_filename = app_metadata['image_filename']
    image_path = app_metadata['image_path']
    prediction = app_metadata['prediction']
    
    scene_graph_figure, _, _, _ = network_graph(image_path, prediction[image_filename])
    return scene_graph_figure

def draw_imagefigure(jsonified_metadata):
    app_metadata = json.loads(jsonified_metadata)
    image_path = app_metadata['image_path']
    h = app_metadata['img_height']
    w = app_metadata['img_width']
    
    img_fig = image_figure(image_path, w, h)
    return img_fig

def draw_bbox_node(jsonified_metadata, img_fig, nodeClickData):
    scale_factor = 0.6
    app_metadata = json.loads(jsonified_metadata)
    bboxes = app_metadata['bboxes']
    h = app_metadata['img_height']
    node_index = int(nodeClickData['id'])
    box = bboxes[node_index]
    img_fig.add_shape(
        type='rect',
        x0=box[0]*scale_factor, x1=box[2]*scale_factor, y0=(h-box[1])*scale_factor, y1=(h-box[3])*scale_factor,
        xref='x', yref='y',
        line_color='#FF007F'
    )
    return img_fig

def highlight_node(scene_graph_figure, nodeClickData):
    node_index = "#" + nodeClickData['id']
    default_stylesheet = scene_graph_figure.stylesheet
    new_styles = [
        {
            'selector': node_index,
            "style": {
                'background-color': '#FF007F',
                "border-width": 2,
                "border-color": "#A6E22E",
                "border-opacity": 1,
                "opacity": 1,
                "label": "data(label)",
                "color": "black",
                "font-size": 20,
                "font-weight":"bold",
                'text-background-color':"#FF007F",
                'text-background-opacity': 0.75,
                'z-index': 9999
            }
        }
    ]
    scene_graph_figure.stylesheet = default_stylesheet + new_styles
    return scene_graph_figure

def draw_bbox_edge(jsonified_metadata, img_fig, edgeClickData):
    scale_factor = 0.6
    app_metadata = json.loads(jsonified_metadata)
    bboxes = app_metadata['bboxes']
    h = app_metadata['img_height']
    edge_index = edgeClickData['id']
    source_index = int(edge_index.split("-")[0])
    target_index = int(edge_index.split("-")[-1])
    
    box = bboxes[source_index]
    img_fig.add_shape(
        type='rect',
        x0=box[0]*scale_factor, x1=box[2]*scale_factor, y0=(h-box[1])*scale_factor, y1=(h-box[3])*scale_factor,
        xref='x', yref='y',
        line_color='#FFA500'
    )

    box = bboxes[target_index]
    img_fig.add_shape(
        type='rect',
        x0=box[0]*scale_factor, x1=box[2]*scale_factor, y0=(h-box[1])*scale_factor, y1=(h-box[3])*scale_factor,
        xref='x', yref='y',
        line_color='#A500FF'
    )

    return img_fig

def highlight_edge(scene_graph_figure, edgeClickData):
    edge_index = edgeClickData['id']
    source_index ="#" + edge_index.split("-")[0]
    target_index = "#" + edge_index.split("-")[-1]
    default_stylesheet = scene_graph_figure.stylesheet

    edge_index = "#" + edge_index
    
    new_styles = [
        {
            'selector': source_index,
            "style": {
                'background-color': '#FFA500',
                "border-width": 2,
                "border-color": "#E6DB74",
                "border-opacity": 1,
                "opacity": 1,
                "label": "data(label)",
                "color": "black",
                "font-size": 20,
                "font-weight":"bold",
                'text-background-color':"#FFA500",
                'text-background-opacity': 0.75,
                'z-index': 9999
            }
        },
        {
            'selector': target_index,
            "style": {
                'background-color': '#A500FF',
                "border-width": 2,
                "border-color": "#E6DB74",
                "border-opacity": 1,
                "opacity": 1,
                "label": "data(label)",
                "color": "black",
                "font-size": 20,
                "font-weight":"bold",
                'text-background-color':"#A500FF",
                'text-background-opacity': 0.75,
                'z-index': 9999
            }
        },
        {
            'selector': edge_index,
            'style': {
                'line-color': '#E6DB74',
                'width':3,
                'target-arrow-color': '#E6DB74',
                "label": "data(label)",
                "color": "black",
                "font-size": 20,
                "font-weight":"bold",
                'text-background-color':"#E6DB74",
                'text-background-opacity': 0.75,
                'z-index': 9999
            }
        }
    ]
    scene_graph_figure.stylesheet = default_stylesheet + new_styles
    return scene_graph_figure

def highlight_dropdown_edges(scene_graph_figure, selected_relation, jsonified_metadata):

    if selected_relation == "NIL":
        return scene_graph_figure

    app_metadata = json.loads(jsonified_metadata)
    pred_rels = app_metadata['predicted_relations']
    default_stylesheet = scene_graph_figure.stylesheet

    edge_ids = []

    for relation in pred_rels:
        rel = relation[1]
        
        if selected_relation == rel:
            source_val = relation[0].split('-')[0]
            target_val = relation[2].split('-')[0]
            if len(rel.split(" ")) == 1:
                edgeID = "#" + source_val + "-" + rel + "-" + target_val
            else:
                edgeID = "#" + source_val + "-" + rel.replace(" ", "_") + "-" + target_val
            edge_ids.append(edgeID)
    
    new_styles = []

    for edgeID in edge_ids:
        new_styles.append(
            {
                'selector': edgeID,
                'style': {
                    'line-color': '#66D9EF',
                    'width':3,
                    'target-arrow-color': '#66D9EF',
                    'z-index': 9999
                }
            }
        )
    
    scene_graph_figure.stylesheet = default_stylesheet + new_styles
    return scene_graph_figure

def filter_confidence_edges(scene_graph_figure, conf_filtering_value, jsonified_metadata):
    
    if conf_filtering_value == 'N':
        return scene_graph_figure
    
    all_elements = scene_graph_figure.elements
    app_metadata = json.loads(jsonified_metadata)
    pred_rels = app_metadata['predicted_relations']
    
    nodes = []
    node_freq_dict = {}

    for element in all_elements:
        if 'source' not in element['data'].keys():
            nodes.append(element)
            node_freq_dict[element['data']['id']] = 0
    
    filter_value = int(conf_filtering_value)
    filtered_edges = []

    #Note: pred_rels and pred_rels_scores are already sorted as per confidence score values
    for relation in pred_rels:
        source_val = relation[0].split("-")[0]

        if node_freq_dict[source_val] < filter_value:
            target_val = relation[2].split("-")[0]
            label_val = relation[0] + " " + relation[1] + " " + relation[2]
            
            if len(relation[1].split(" ")) == 1: 
                id_val = source_val + "-" + relation[1] + "-" + target_val
            else:
                id_val = source_val + "-" + relation[1].replace(" ", "_") + "-" + target_val
            
            edge = {'data': {'id': id_val, 'source': source_val, 'target': target_val, 'label': label_val}}
            filtered_edges.append(edge)
            node_freq_dict[source_val] = node_freq_dict[source_val] + 1
    
    scene_graph_figure.elements = nodes + filtered_edges
    return scene_graph_figure

def abstract_labels(scene_graph_figure, hypernym_abs_val, jsonified_metadata):
    if hypernym_abs_val == 1:
        return scene_graph_figure

    app_metadata = json.loads(jsonified_metadata)
    hypernyms = app_metadata['hypernyms']
    all_elements = scene_graph_figure.elements

    for label in hypernyms.keys():
        hypernyms[label] = hypernyms[label][:hypernym_abs_val]
        hypernyms[label] = hypernyms[label][-1]
    

    #Changing Node labels as per abstraction slider level using hypernyms
    nodes = []
    label_change_dict = {}
    for element in all_elements:
        if 'source' not in element['data'].keys():
            element_val = element
            old_label = element_val['data']['label'].split("-")[1]
            new_label = ""
            if old_label in hypernyms.keys():
                new_label = element_val['data']['label'].split("-")[0] + "-" + hypernyms[old_label]
                label_change_dict[element_val['data']['id']] = new_label
            else:
                new_label = element_val['data']['label']
            element_val['data']['label'] = new_label
            nodes.append(element_val)
    

    #Changing Edge Labels as per abstraction slider level using hypernyms
    edges = []
    for element in all_elements:
        if 'source' in element['data'].keys():
            element_val = element
            source_id = element_val['data']['source']
            target_id = element_val['data']['target']
            relation_label = element_val['data']['label'].split(" ")[1]
            source_label = ""
            target_label = ""

            if source_id in label_change_dict.keys():
                source_label = label_change_dict[source_id]
            else:
                source_label = element_val['data']['label'].split(" ")[0]
            
            if target_id in label_change_dict.keys():
                target_label = label_change_dict[target_id]
            else:
                target_label = element_val['data']['label'].split(" ")[2]
            
            element_val['data']['label'] = source_label + " " + relation_label + " " + target_label

            edges.append(element_val)
    
    scene_graph_figure.elements = nodes + edges
    
    return scene_graph_figure
        
    
def return_relation_table(jsonified_metadata, nodeClickData):
    app_metadata = json.loads(jsonified_metadata)
    pred_rels = app_metadata['predicted_relations']
    pred_rels_scores = app_metadata['pred_rel_scores']
    
    nodeNumber = nodeClickData['id']
    node_details = {
        'node_text': nodeClickData['label'],
        'relations': []
    }
    for rel_index, relation in enumerate(pred_rels):
        subject = relation[0]
        predicate = relation[2]
        subject_index = subject.split("-")[0]
        predicate_index = predicate.split("-")[0]
        if nodeNumber == subject_index or nodeNumber == predicate_index:
            node_details['relations'].append({"Subject": subject, "Relation": relation[1], "Predicate": predicate, "Confidence Score": pred_rels_scores[rel_index]})
    
    relations_dataframe = pd.DataFrame(node_details['relations'])

    rel_table = dash_table.DataTable(
        id='relation-table',
        columns=[{"name": i, "id": i} for i in relations_dataframe.columns],
        data=relations_dataframe.to_dict('records'),
    )

    if len(node_details['relations']) > 0:
        table_title = node_details['node_text'] + " has " + str(len(node_details['relations'])) + " total number of relations. Relation Details for Node " + node_details['node_text'] + " are as follows."
    else:
        table_title = "Node " + node_details['node_text'] + " doesn't have any relations."

    return [table_title, rel_table]


def return_dash_app():
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    scale_factor = 0.6
    colors = {
        'background': '#D3D3D3',
        'text': '#000000'
    }
    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll',
            'font-family': 'verdana'
        }
    }
    
    app.layout = html.Div(
        style={
            'backgroundColor': colors['background'], 
            'position': 'fixed', 
            'width': '100%', 
            'height': '100%', 
            'top': '0px', 
            'left': '0px', 
            'overflow-y': 'scroll',
            'overflow-x': 'scroll'
        }, 
        children=[
            html.H1(
                children='Scene Graph Visualisation',
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                    'font-family':'verdana'
                }
            ),
            html.P(
                children=[
                    'NOTE: On selecting Top n Most Confident relations, n edges with highest confidence score per source node will be displayed',
                    html.Br(),
                    'NOTE: Relation table is not affected by Top N Confidence Dropdown, and displays all relaionships a node is part of as both a Subject and a Predicate.'
                ],
                style={
                    'font-family':'verdana',
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),

            #To Upload Input Image
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
            ),
                        
            html.Div(
                id = "Graph1",
                style={
                    'float':'left', 
                    'width':'50%', 
                },
                children=[
                    cyto.Cytoscape(
                        id='cytoscape-scenegraph',
                        elements=[]
                    )
                ]
            ),

            html.Div(
                style = {
                    'float': 'left',
                    'width': '50%',
                },
                children=[
                    dcc.Graph(
                        id='Image1'
                    )
                ]
            ),

            html.Div(
                style = {
                    'float':'left', 
                    'width':'50%',
                    'font-family':'verdana'
                },
                children = [
                    'Select Level of Abstraction for Node Labels (Hypernymy)',
                    dcc.Slider(
                        id='hypernym-slider',
                        min=1,
                        max=5,
                        step=1,
                        value=1,
                    )
                ]
            ),

            html.Div(
                style = {
                    'float':'left', 
                    'width':'50%',
                    'font-family':'verdana'
                },
                children = [
                    'Select the Type of Relationship you want to highlight in the graph.',
                    dcc.Dropdown(
                        id='relation-dropdown',
                        options=[
                            {'label': 'None', 'value': 'NIL'},
                        ],
                        value='NIL'
                    )
                ]
            ),

            html.Div(
                style = {
                    'float':'left', 
                    'width':'50%',
                    'font-family':'verdana'
                },
                children = [
                    'Select the relations you want to filter out based on their prediction Confidence scores',
                    dcc.Dropdown(
                        id='confidence-dropdown',
                        options=[
                            {'label': 'Top 1 Most Confident', 'value': '1'},
                            {'label': 'Top 2 Most Confident', 'value': '2'},
                            {'label': 'Top 3 Most Confident', 'value': '3'},
                            {'label': 'Top 5 Most Confident', 'value': '5'},
                            {'label': 'Top 10 Most Confident', 'value': '10'},
                            {'label': 'Top 15 Most Confident', 'value': '15'},
                            {'label': 'All Relations', 'value': 'N'}
                        ],
                        value='1'
                    )
                ]
            ),

            #Node details from the Graph are displayed here
            html.Div(
                style = {
                    'float':'left', 
                    'width':'50%',
                    'font-family':'verdana'
                },
                children = [
                    'Click over Nodes in the graph to display relation details. ',
                    html.Div(
                        id='relation-data'
                    )
                ], 
            ),

            html.Div(
                id='app-metadata', 
                style={'display': 'none'},
            ),

            html.Div(id = 'search_results'),
            # Placeholder for the selected image URL
            html.Div(id='selected-image-url', style={'margin-top': '20px'})
    ])

    def extract_information(input_string):
    # Define the regular expression pattern
        pattern = re.compile(r'(\d+)-(\w+)\s+([a-zA-Z\s]+)\s+(\d+)-(\w+)')

        # Match the pattern in the input string
        match = pattern.match(input_string)

        if match:
            # Extract information from the matched groups
            first_building = match.group(2)
            relationship = match.group(3)
            second_building = match.group(5)

            return first_building, relationship, second_building
        else:
            # Return None if no match is found
            return None


    #To Draw the predicted scene Graph based on the input image
    @app.callback( [Output('app-metadata', 'children'), Output('relation-dropdown', 'options'),
    Output('relation-dropdown', 'value'), Output('confidence-dropdown', 'value'), Output('hypernym-slider', 'value')],
            [
                Input('upload-data', 'contents'),
                Input('upload-data', 'filename')
            ])
    def update_metadata(contents, filename):
        
        if filename is not None and contents is not None:
            image_path = os.path.join(UPLOAD_DIRECTORY, filename)
            pic = Image.open(image_path)

            resized_y, resized_x = pic.size

            result = subprocess.run(["python", "./sgg_model/generate_sgg.py", "--img_path", filename, "--resume ./sgg_model/ckpt/checkpoint0149.pth"], capture_output=True, text=True)


            #Reading the predictions saved by the model from the JSON file
            output_filename = "output_sgg.json"
            output_path = os.path.join("../data/scene_graphs/", output_filename)
            prediction = []
            with open(output_path, 'r') as f:
                prediction = json.load(f)
            
            selected_prediction = {}
            selected_prediction[filename] = prediction
            prediction = selected_prediction
            
            #Creating the network graph using the predictions
            _, bboxes, pred_rels, pred_rel_scores = network_graph(image_path, prediction[filename])
            
            #Loading the image size proportionate to the bounding boxes as per the preprocessed image
            original_y, original_x = return_dimensions(bboxes)
            #Scale as per which the bounding boxes must be scaled
            x_scale = resized_x/original_x
            y_scale = resized_y/original_y

            scaled_bboxes = []
            for i in range(0, len(bboxes)):            
                #Scaling the bounding boxes
                x1 = int(np.round(bboxes[i][0]*x_scale))
                y1 = int(np.round(bboxes[i][1]*y_scale))
                x2 = int(np.round(bboxes[i][2]*x_scale))
                y2 = int(np.round(bboxes[i][3]*y_scale))

                scaled_bboxes.append([x1, y1, x2, y2])
            
            #Sorting Precited Relations based on their confidence scores.
            combined_rel_confidence = zip(pred_rel_scores, pred_rels)
            combined_rel_confidence = sorted(combined_rel_confidence, key=lambda x: x[0], reverse=True)
            combined_rel_confidence = list(combined_rel_confidence) 
            pred_rel_scores, pred_rels = zip(*combined_rel_confidence)
            
            
            #Loading Hypernyms and Meronyms
            semantic_info = {}
            with open("./data/semantic_filtering/semantic_info.json", "r") as f:
                semantic_info = json.load(f)
            

            metadata_dict = {
                'image_filename': filename,
                'image_path': image_path,
                'bboxes': scaled_bboxes,
                'predicted_relations': pred_rels,
                'pred_rel_scores': pred_rel_scores,
                'img_width': resized_y,
                'img_height': resized_x,
                'prediction': prediction,
                'meronyms': semantic_info['meronyms'],
                'hypernyms': semantic_info['hypernyms']
            }

            rel_dropdown_options = load_dropdown_options(pred_rels)
            rel_dropdown_default_val = "NIL"
            confidence_default_val = "1"
            hypernym_slider_default = 1

            return json.dumps(metadata_dict), rel_dropdown_options, rel_dropdown_default_val, confidence_default_val, hypernym_slider_default
        
        else:
            return no_update, no_update, no_update, no_update, no_update
      
        
    @app.callback([Output('search_results', 'children')], [Input('cytoscape-scenegraph', 'tapNodeData'), Input('cytoscape-scenegraph', 'tapEdgeData')])
    def show_search_results(nodeData, edgeData):
        
        if nodeData is not None:
            selected_node_for_search = nodeData['label'].split("-")[-1]
            search_img_ids = search_nodes(selected_node_for_search)

            image_urls = ["https://cs.stanford.edu/people/rak248/VG_100K_2/" + str(image_id) + ".jpg" for image_id in search_img_ids]

            # Use dcc.Image component for each image
            search_results = [html.Img(
                id=f'image-{index}',
                src=image_url,
                style={'width': '100px', 'height': '100px'},
                # Add a click event to capture the image URL
                n_clicks=0
            )
            for index, image_url in enumerate(image_urls)]


            return [search_results]
        elif edgeData is not None:
            print(edgeData['label'])
            selected_subject_for_search, selected_relation_for_search, selected_object_for_search = extract_information(edgeData['label'])

            print(selected_subject_for_search)
            print(selected_relation_for_search)
            print(selected_object_for_search)

            print("Searching")
            rel_search_img_ids = search_edges(selected_subject_for_search, selected_relation_for_search, selected_object_for_search)
            print("Search finished.")

            print(len(rel_search_img_ids))


            rel_image_urls = ["https://cs.stanford.edu/people/rak248/VG_100K_2/" + str(image_id) + ".jpg" for image_id in rel_search_img_ids]

            print(rel_image_urls)

            # Use dcc.Image component for each image
            rel_search_results = [html.Img(
                id=f'image-{index}',
                src=image_url,
                style={'width': '100px', 'height': '100px'},
                # Add a click event to capture the image URL
                n_clicks=0
            )
            for index, image_url in enumerate(rel_image_urls)]

            print(rel_search_results)

            return rel_search_results
        else:
            return no_update
        
      


    @app.callback([Output('Graph1', 'children'), Output('relation-data', 'children'), 
    Output('Image1', 'figure')],
    [Input('app-metadata', 'children'), Input('cytoscape-scenegraph', 'tapNodeData'), 
    Input('cytoscape-scenegraph', 'tapEdgeData'), Input('confidence-dropdown', 'value'), Input('relation-dropdown', 'value'), Input('hypernym-slider', 'value')])
    def general_update_app(jsonified_metadata, nodeClickData, edgeClickData, confDropDownData, relDropDownData, hypernymSliderVal):
        if jsonified_metadata is not None:
            scene_graph_figure = draw_scenegraph(jsonified_metadata)
            image_figure = draw_imagefigure(jsonified_metadata)

            conf_filtering_value = confDropDownData
            scene_graph_figure = filter_confidence_edges(scene_graph_figure, conf_filtering_value, jsonified_metadata)
            
            hypernym_abs_val = hypernymSliderVal
            scene_graph_figure = abstract_labels(scene_graph_figure, hypernym_abs_val, jsonified_metadata)

            selected_relation = relDropDownData
            scene_graph_figure = highlight_dropdown_edges(scene_graph_figure, selected_relation, jsonified_metadata)
        
            if nodeClickData is not None:
                
                image_figure = draw_bbox_node(jsonified_metadata, image_figure, nodeClickData)
                scene_graph_figure = highlight_node(scene_graph_figure, nodeClickData)
                relation_data = return_relation_table(jsonified_metadata, nodeClickData)

                return scene_graph_figure, relation_data, image_figure
            
            elif edgeClickData is not None:

                image_figure = draw_bbox_edge(jsonified_metadata, image_figure, edgeClickData)
                scene_graph_figure = highlight_edge(scene_graph_figure, edgeClickData)

                return scene_graph_figure, "", image_figure
                
            else:
                return scene_graph_figure, "", image_figure
        
        else:
            return no_update, no_update, no_update
          

    return app

    

def main():
    dash_app = return_dash_app()
    dash_app.run_server(debug=True)


if __name__ == '__main__':
    main()