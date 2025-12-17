"""
Interactive Dash application to visualize the 4 classification models
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Load Atkinson font
FONT_PATH = Path(__file__).parent / "fonts" / "AtkinsonHyperlegible-Regular.ttf"
FONT_FAMILY = "Atkinson Hyperlegible, Arial, sans-serif"

# Configuration des couleurs pour chaque modèle
MODEL_COLORS = {
    "AACnnClassifier": "#4ECDC4",  # Turquoise
    "AACnnClassifier_Run2": "#2E9B95",  # Turquoise foncé pour Run 2
    "DenseMsaClassifier": "#96CEB4",  # Vert clair
    "DenseSiteClassifier": "#FFEAA7",  # Jaune
    "LogisticRegressionClassifier": "#FF6B6B",  # Rouge
}

# Données des modèles
MODELS_DATA = {
    "AACnnClassifier": {
        "type": "CNN",
        "architecture": "Conv1D - ReLU - AdaptiveAvgPool - Dropout(0.2) - Linear",
        "layers": ["Conv1d(21-64)", "ReLU", "AdaptiveAvgPool1d", "Dropout(0.2)", "Linear(64-1)"],
        "n_params": 1537,
        "data_type": "SiteCompositionData",
        "input_shape": "(batch, sites, 21)",
        "f1_score": 0.838,
        "val_acc": 0.843,
        "val_loss": 0.548,
        "best_epoch": 363,
        "training_time": 393,
        "hyperparams": {
            "batch_size": 64,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            "kernel_size": 1,
            "dropout": 0.2
        }
    },
    "DenseMsaClassifier": {
        "type": "Dense Neural Network",
        "architecture": "Linear - BatchNorm - ReLU - Dropout - Linear - BatchNorm - ReLU - Dropout - Linear",
        "layers": ["Linear(21-128)", "BatchNorm", "ReLU", "Dropout(0.4)", 
                  "Linear(128-64)", "BatchNorm", "ReLU", "Dropout(0.3)", "Linear(64-1)"],
        "n_params": 11649,
        "data_type": "MsaCompositionData",
        "input_shape": "(batch, 21)",
        "f1_score": 0.537,
        "val_acc": 0.640,
        "val_loss": 0.660,
        "best_epoch": 8,
        "training_time": 16,
        "hyperparams": {
            "batch_size": 64,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            "dropout_1": 0.4,
            "dropout_2": 0.3
        }
    },
    "DenseSiteClassifier": {
        "type": "Dense Neural Network",
        "architecture": "Flatten - Linear - BatchNorm - ReLU - Dropout - Linear - BatchNorm - ReLU - Dropout - Linear",
        "layers": ["Flatten(sites×21)", "Linear(-128)", "BatchNorm", "ReLU", "Dropout(0.4)",
                  "Linear(128-64)", "BatchNorm", "ReLU", "Dropout(0.3)", "Linear(64-1)"],
        "n_params": 5153665,
        "data_type": "SiteCompositionData",
        "input_shape": "(batch, sites, 21)",
        "f1_score": 0.0,
        "val_acc": 0.529,
        "val_loss": 0.693,
        "best_epoch": 1,
        "training_time": 26,
        "hyperparams": {
            "batch_size": 64,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            "dropout_1": 0.4,
            "dropout_2": 0.3
        }
    },
    "LogisticRegressionClassifier": {
        "type": "Logistic Regression",
        "architecture": "Linear Classifier",
        "layers": ["Linear(21-1)", "Sigmoid"],
        "n_params": 22,
        "data_type": "MsaCompositionData",
        "input_shape": "(batch, 21)",
        "f1_score": 0.410,
        "val_acc": 0.380,
        "val_loss": None,  # Logistic regression uses cross-validation, no single val_loss
        "best_epoch": None,
        "training_time": 1,
        "hyperparams": {
            "cv": 50,
            "scale_features": True,
            "shuffle_data": True
        }
    },
    "AACnnClassifier_Run2": {
        "type": "CNN (Run 2 - Best Model)",
        "architecture": "Conv1D - ReLU - AdaptiveAvgPool - Dropout(0.2) - Linear",
        "layers": ["Conv1d(21-64)", "ReLU", "AdaptiveAvgPool1d", "Dropout(0.2)", "Linear(64-1)"],
        "n_params": 1537,
        "data_type": "SiteCompositionData",
        "input_shape": "(batch, sites, 21)",
        "f1_score": 0.758,
        "val_acc": 0.798,
        "val_loss": 0.564,
        "best_epoch": 329,
        "training_time": 757,
        "hyperparams": {
            "batch_size": 64,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            "kernel_size": 1,
            "dropout": 0.2
        }
    }
}

def format_number(num):
    """Format a number for display"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(int(num))

def format_training_time(seconds):
    """Format training time in seconds to min:sec if > 60s, else just seconds"""
    if seconds >= 60:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        return f"{seconds}s"

def create_architecture_diagram(model_name):
    """Create a Mermaid-like graph diagram for a model architecture"""
    model_data = MODELS_DATA[model_name]
    color = MODEL_COLORS[model_name]
    layers = model_data["layers"]
    
    fig = go.Figure()
    
    # Create nodes (layers) positioned horizontally
    n_layers = len(layers)
    x_positions = np.linspace(0.15, 0.85, n_layers)
    y_center = 0.5
    node_width = 0.12
    node_height = 0.15
    
    # First, draw all arrows (so they appear behind nodes)
    for i in range(n_layers - 1):
        x_start = x_positions[i] + node_width/2
        x_end = x_positions[i+1] - node_width/2
        
        # Draw arrow using annotation (omit axref/ayref - Plotly will infer from xref/yref)
        fig.add_annotation(
            x=x_end, y=y_center,
            ax=x_start, ay=y_center,
            xref="paper", yref="paper",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.8,
            arrowwidth=3,
            arrowcolor=color
        )
    
    # Then draw nodes (so they appear on top)
    for i, layer in enumerate(layers):
        x_pos = x_positions[i]
        
        # Add node as a rounded rectangle
        fig.add_shape(
            type="rect",
            x0=x_pos - node_width/2, y0=y_center - node_height/2,
            x1=x_pos + node_width/2, y1=y_center + node_height/2,
            fillcolor=color,
            opacity=0.25,
            line=dict(color=color, width=3),
            xref="paper", yref="paper",
            layer="above"
        )
        
        # Add layer text
        fig.add_annotation(
            x=x_pos, y=y_center,
            text=f"<b>{layer}</b>",
            showarrow=False,
            font=dict(size=10, color="black"),
            xref="paper", yref="paper",
            bgcolor="white",
            bordercolor=color,
            borderwidth=2,
            borderpad=6
        )
    
    fig.update_layout(
        title=dict(text=f"Architecture Diagram - {model_name}", font=dict(family=FONT_FAMILY)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        height=250,
        margin=dict(l=50, r=50, t=60, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT_FAMILY)
    )
    
    return fig

def create_comparison_chart():
    """Create a comparison chart of models"""
    model_names = list(MODELS_DATA.keys())
    f1_scores = [MODELS_DATA[m]["f1_score"] for m in model_names]
    val_accs = [MODELS_DATA[m]["val_acc"] for m in model_names]
    colors_list = [MODEL_COLORS[m] for m in model_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[m.replace('Classifier', '') for m in model_names],
        y=f1_scores,
        name='F1 Score',
        marker_color=colors_list,
        text=[f'{s:.3f}' for s in f1_scores],
        textposition='outside',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=[m.replace('Classifier', '') for m in model_names],
        y=val_accs,
        name='Validation Accuracy',
        marker_color=colors_list,
        text=[f'{a:.3f}' for a in val_accs],
        textposition='outside',
        opacity=0.5
    ))
    
    fig.update_layout(
        title=dict(text="Performance Comparison", font=dict(family=FONT_FAMILY)),
        xaxis_title=dict(text="Models", font=dict(family=FONT_FAMILY)),
        yaxis_title=dict(text="Score", font=dict(family=FONT_FAMILY)),
        barmode='group',
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family=FONT_FAMILY)),
        font=dict(family=FONT_FAMILY)
    )
    
    return fig

def create_loss_chart():
    """Create a comparison chart of validation loss"""
    model_names = list(MODELS_DATA.keys())
    val_losses = [MODELS_DATA[m]["val_loss"] for m in model_names]
    colors_list = [MODEL_COLORS[m] for m in model_names]
    
    # Filter out None values for LogisticRegression
    valid_indices = [i for i, loss in enumerate(val_losses) if loss is not None]
    valid_models = [model_names[i] for i in valid_indices]
    valid_losses = [val_losses[i] for i in valid_indices]
    valid_colors = [colors_list[i] for i in valid_indices]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[m.replace('Classifier', '') for m in valid_models],
        y=valid_losses,
        marker_color=valid_colors,
        text=[f'{l:.3f}' for l in valid_losses],
        textposition='outside',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=dict(text="Validation Loss Comparison (Lower is Better)", font=dict(family=FONT_FAMILY)),
        xaxis_title=dict(text="Models", font=dict(family=FONT_FAMILY)),
        yaxis_title=dict(text="Validation Loss", font=dict(family=FONT_FAMILY)),
        height=400,
        template="plotly_white",
        font=dict(family=FONT_FAMILY)
    )
    
    return fig

def create_params_chart():
    """Create a comparison chart of number of parameters"""
    model_names = list(MODELS_DATA.keys())
    n_params = [MODELS_DATA[m]["n_params"] for m in model_names]
    colors_list = [MODEL_COLORS[m] for m in model_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[m.replace('Classifier', '') for m in model_names],
        y=n_params,
        marker_color=colors_list,
        text=[format_number(p) for p in n_params],
        textposition='outside',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=dict(text="Model Complexity (Number of Parameters)", font=dict(family=FONT_FAMILY)),
        xaxis_title=dict(text="Models", font=dict(family=FONT_FAMILY)),
        yaxis_title=dict(text="Number of Parameters", font=dict(family=FONT_FAMILY)),
        yaxis_type="log",
        height=400,
        template="plotly_white",
        font=dict(family=FONT_FAMILY)
    )
    
    return fig

# Initialize Dash application
app = dash.Dash(__name__)
app.title = "Classification Models - Dashboard"

# Add custom CSS for Atkinson font
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @font-face {
                font-family: 'Atkinson Hyperlegible';
                src: url('assets/fonts/AtkinsonHyperlegible-Regular.ttf') format('truetype');
                font-weight: normal;
                font-style: normal;
            }
            @font-face {
                font-family: 'Atkinson Hyperlegible';
                src: url('assets/fonts/AtkinsonHyperlegible-Bold.ttf') format('truetype');
                font-weight: bold;
                font-style: normal;
            }
            @font-face {
                font-family: 'Atkinson Hyperlegible';
                src: url('assets/fonts/AtkinsonHyperlegible-Italic.ttf') format('truetype');
                font-weight: normal;
                font-style: italic;
            }
            * {
                font-family: 'Atkinson Hyperlegible', Arial, sans-serif !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Application layout
app.layout = html.Div([
    html.Div([
        html.H1("Classification Models Architecture & Parameters", 
                style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#1f77b4', 'fontFamily': FONT_FAMILY}),
    ]),
    
    # Model selector
    html.Div([
        html.Label("Select a model:", style={'fontSize': '18px', 'fontWeight': 'bold', 'fontFamily': FONT_FAMILY}),
        dcc.Dropdown(
            id='model-selector',
            options=[{'label': name, 'value': name} for name in MODELS_DATA.keys()],
            value=list(MODELS_DATA.keys())[0],
            style={'width': '100%', 'marginBottom': '20px', 'fontFamily': FONT_FAMILY}
        ),
    ], style={'width': '100%', 'padding': '20px'}),
    
    # Selected model information
    html.Div(id='model-info', style={'padding': '20px'}),
    
    # Architecture diagram
    html.Div([
        dcc.Graph(id='architecture-diagram')
    ], style={'padding': '20px'}),
    
    # Comparison charts
    html.Div([
        html.H2("Comparisons", style={'textAlign': 'center', 'marginTop': '40px', 'fontFamily': FONT_FAMILY}),
        html.Div([
            html.Div([
                dcc.Graph(id='performance-chart', figure=create_comparison_chart())
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                dcc.Graph(id='loss-chart', figure=create_loss_chart())
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                dcc.Graph(id='params-chart', figure=create_params_chart())
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'center'})
    ]),
    
], style={'fontFamily': FONT_FAMILY, 'maxWidth': '1400px', 'margin': '0 auto'})

@callback(
    [Output('model-info', 'children'),
     Output('architecture-diagram', 'figure')],
    [Input('model-selector', 'value')]
)
def update_model_info(selected_model):
    """Update information for the selected model"""
    model_data = MODELS_DATA[selected_model]
    color = MODEL_COLORS[selected_model]
    
    # Create information cards
    info_cards = html.Div([
        html.Div([
            html.H3(selected_model, style={'color': color, 'marginBottom': '10px', 'fontFamily': FONT_FAMILY}),
            html.P(f"Type: {model_data['type']}", style={'fontSize': '16px', 'fontStyle': 'italic', 'fontFamily': FONT_FAMILY}),
            html.Hr(),
            html.Div([
                html.Div([
                    html.H4("Parameters", style={'color': color, 'fontFamily': FONT_FAMILY}),
                    html.P(f"Number of parameters: {format_number(model_data['n_params'])}", style={'fontFamily': FONT_FAMILY}),
                    html.P(f"Data type: {model_data['data_type']}", style={'fontFamily': FONT_FAMILY}),
                    html.P(f"Input shape: {model_data['input_shape']}", style={'fontFamily': FONT_FAMILY}),
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                html.Div([
                    html.H4("Loss Function", style={'color': color, 'fontFamily': FONT_FAMILY}),
                    html.P(f"Validation Loss: {model_data['val_loss']:.3f}" if model_data['val_loss'] is not None else "Validation Loss: N/A (CV)", 
                           style={'color': 'darkred', 'fontWeight': 'bold', 'fontFamily': FONT_FAMILY}),
                    html.P(f"Best Epoch: {model_data['best_epoch'] if model_data['best_epoch'] else 'N/A'}", style={'fontFamily': FONT_FAMILY}),
                    html.P(f"Training Time: {format_training_time(model_data['training_time'])}", style={'fontFamily': FONT_FAMILY}),
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                html.Div([
                    html.H4("Performance", style={'color': color, 'fontFamily': FONT_FAMILY}),
                    html.P(f"F1 Score: {model_data['f1_score']:.3f}", style={'color': 'darkgreen', 'fontWeight': 'bold', 'fontFamily': FONT_FAMILY}),
                    html.P(f"Validation Accuracy: {model_data['val_acc']:.3f}", style={'color': 'darkblue', 'fontWeight': 'bold', 'fontFamily': FONT_FAMILY}),
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                html.Div([
                    html.H4("Hyperparameters", style={'color': color, 'fontFamily': FONT_FAMILY}),
                    html.Ul([
                        html.Li(f"{k}: {v}", style={'fontFamily': FONT_FAMILY}) for k, v in model_data['hyperparams'].items()
                    ])
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),
        ], style={
            'border': f'3px solid {color}',
            'borderRadius': '10px',
            'padding': '20px',
            'backgroundColor': f'{color}15',
            'marginBottom': '20px',
            'fontFamily': FONT_FAMILY
        })
    ])
    
    # Create architecture diagram
    arch_fig = create_architecture_diagram(selected_model)
    
    return info_cards, arch_fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)

