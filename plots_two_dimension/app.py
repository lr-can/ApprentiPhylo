import os
import json
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import re
import logging
import plotly.colors as pc

# Set up logging to file
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Classifier and marker mapping
CLASSIFIERS = [
    'AACnnClassifier',
    'DenseMsaClassifier',
    'DenseSiteClassifier',
    'LogisticRegressionClassifier',
]
CLASSIFIER_MARKERS = {
    'AACnnClassifier': 'circle',
    'DenseMsaClassifier': 'triangle-down',
    'DenseSiteClassifier': 'square',
    'LogisticRegressionClassifier': 'x',
}

# Possible suffix orders for different simulation types
SUFFIX_ORDERS = [
    ['E_0', 'E_0.05', 'E_0.1', 'E_0.2', 'E_0.5'],
    ['R_0', 'R_0.05', 'R_0.1', 'R_0.2', 'R_0.5'],
]

# Suffix to marker size mapping
SUFFIX_MARKER_SIZE = {
    'E_0': 10,
    'E_0.05': 12,
    'E_0.1': 14,
    'E_0.2': 16,
    'E_0.5': 18,
    'R_0': 10,
    'R_0.05': 12,
    'R_0.1': 14,
    'R_0.2': 16,
    'R_0.5': 18,
}
# Group to marker shape mapping
GROUP_MARKERS = {}
MARKER_SEQ = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'x']
COLOR_SEQ = pc.qualitative.Plotly

# Group color mapping
GROUP_COLORS = {}

# Group to marker shape mapping
GROUP_MARKERS = {}
MARKER_SEQ = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'x']


GROUP_ROOT = '../results/MPD/viridiplantae_group_results'
RUNS_ROOT = '../results/simulations-classifiers/runs_viridiplantae'

# Get all available groups
def get_available_groups():
    abs_group_root = os.path.abspath(GROUP_ROOT)
    if not os.path.exists(GROUP_ROOT):
        return []
    # Only return directories starting with 'GROUP'
    return [d for d in os.listdir(GROUP_ROOT) if os.path.isdir(os.path.join(GROUP_ROOT, d)) and d.startswith('GROUP')]

# Extract suffix for sorting (e.g., ext_0.05, root_0.1, etc.)
def extract_suffix(sim_name):
    m = re.search(r'(E|R)_[0-9.]+$', sim_name)
    return m.group(0) if m else sim_name

def extract_suffix_value(suffix):
    # Extract the numeric part from E_0.05 or R_0.2 for sorting
    import re
    m = re.search(r'([ER])_([0-9.]+)', suffix)
    if m:
        return float(m.group(2))
    return 999

def get_suffix_order(sim_names):
    # Try to match the best order for the current simulation names
    for order in SUFFIX_ORDERS:
        if all(any(suf in n for n in sim_names) for suf in order):
            return order
    # fallback: sort by appearance
    return sorted(list({extract_suffix(n) for n in sim_names}))

# Collect all data into a DataFrame
# Recursively search for all distance_results_*.csv, sim_name is the parent directory name

def collect_all_data():
    records = []
    available_groups = get_available_groups()
    
    if not available_groups:
        logging.error(f"No groups found in {GROUP_ROOT}")
        return pd.DataFrame()
        
    for group in available_groups:
        group_path = os.path.join(GROUP_ROOT, group)
        if not os.path.isdir(group_path):
            continue
        for root, dirs, files in os.walk(group_path):
            for file in files:
                if file.startswith('distance_results') and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    distance = None
                    try:
                        df_dist = pd.read_csv(file_path)
                        distance = df_dist['Average_Distance'].mean()
                    except Exception as e:
                        logging.error(f"Error reading {file_path}: {str(e)}")
                        continue
                    if distance is None:
                        continue
                    sim_name = os.path.basename(os.path.dirname(file_path))
                    subgroup = extract_suffix(sim_name)
                    # Only collect simulation folder with the exact same name as sim_name
                    sim_folder = os.path.join(RUNS_ROOT, sim_name)
                    if not os.path.isdir(sim_folder):
                        continue
                    for clf in CLASSIFIERS:
                        clf_folder = os.path.join(sim_folder, clf)
                        summary_path = os.path.join(clf_folder, 'summary.json')
                        if os.path.isfile(summary_path):
                            try:
                                with open(summary_path, 'r') as f:
                                    data = json.load(f)
                                if clf == 'LogisticRegressionClassifier':
                                    accs = data.get('fold_accuracies', [])
                                    accs = [float(a) for a in accs]
                                    acc = sum(accs) / len(accs) if accs else None
                                else:
                                    acc = data.get('val_acc', None)
                                logging.info(f'group={group}, sim_name={sim_name}, clf={clf}, acc={acc}')
                                if acc is not None:
                                    label = f'{group}/{sim_name}/{clf}'
                                    records.append({
                                        'group': group,
                                        'subgroup': subgroup,
                                        'sim_name': sim_name,
                                        'classifier': clf,
                                        'distance': distance,
                                        'acc': acc,
                                        'label': label
                                    })
                            except Exception as e:
                                logging.error(f"Error processing {summary_path}: {str(e)}")
                                continue
    df = pd.DataFrame(records)
    return df

# Initialize data
DATA_DF = collect_all_data()

# Create Dash application
app = Dash(__name__)
app.title = '2D Distance-Classifier Interactive Visualization'

# Get available groups
available_groups = sorted(DATA_DF['group'].unique()) if not DATA_DF.empty else []

# UI layout
app.layout = html.Div([
    html.H2('2D Distance-Classifier Interactive Visualization', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label('Select Group(s)'),
            dcc.Dropdown(
                id='group-dropdown',
                options=[{'label': g, 'value': g} for g in available_groups],
                value=available_groups[:1] if available_groups else [],
                multi=True,
                style={'width': '90%', 'zIndex': 1000, 'margin': '0 auto'}
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%', 'textAlign': 'center'}),
        html.Div([
            html.Label('Select Classifier(s)'),
            dcc.Dropdown(
                id='clf-dropdown',
                options=[{'label': c, 'value': c} for c in CLASSIFIERS],
                value=list(CLASSIFIERS),
                multi=True,
                style={'width': '90%', 'zIndex': 999, 'margin': '0 auto'}
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%', 'textAlign': 'center'}),
        html.Div([
            html.Label('Show subgroup:'),
            dcc.Checklist(
                id='data2-check',
                options=[
                    {'label': 'With data2', 'value': 'with_data2'},
                    {'label': 'Without data2', 'value': 'without_data2'}
                ],
                value=['with_data2', 'without_data2'],
                inline=True,
                style={'marginTop': '8px', 'width': '90%', 'margin': '0 auto'}
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%', 'textAlign': 'center'}),
        html.Div([
            html.Label('Connect by line (ordered by simulation suffix)'),
            dcc.Checklist(
                id='line-check',
                options=[{'label': 'Connect', 'value': 'line'}],
                value=[],
                style={'marginTop': '8px', 'width': '90%', 'margin': '0 auto'}
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%', 'textAlign': 'center'}),
    ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '10px', 'width': '100%'}),
    html.Div([
        dcc.Graph(id='main-plot', style={'height': '700px'}),
    ], style={'width': '100%', 'display': 'block', 'verticalAlign': 'top'}),
    html.Div(id='point-label', style={'textAlign': 'center', 'fontSize': 18, 'color': 'blue', 'marginTop': '20px'})
])

# Main plot callback
@app.callback(
    Output('main-plot', 'figure'),
    [Input('group-dropdown', 'value'),
     Input('clf-dropdown', 'value'),
     Input('line-check', 'value'),
     Input('data2-check', 'value')]
)
def update_plot(selected_groups, selected_clfs, line_check, data2_check):
    if DATA_DF.empty:
        return go.Figure()
        
    df = DATA_DF.copy()
    if selected_groups:
        df = df[df['group'].isin(selected_groups)]
    if selected_clfs:
        df = df[df['classifier'].isin(selected_clfs)]
    # Strictly distinguish with_data2/without_data2
    def is_with_data2(sim_name):
        return 'data2' in sim_name.lower()
    show_with = 'with_data2' in data2_check
    show_without = 'without_data2' in data2_check
    # Only keep user-selected subgroups
    df = df[df['sim_name'].apply(lambda x: (is_with_data2(x) and show_with) or (not is_with_data2(x) and show_without))]
    fig = go.Figure()
    # Assign group-marker
    group_list = sorted(df['group'].unique())
    for i, g in enumerate(group_list):
        GROUP_MARKERS[g] = MARKER_SEQ[i % len(MARKER_SEQ)]
    # Assign classifier colors
    clf_list = sorted(df['classifier'].unique())
    CLASSIFIER_COLORS = {c: COLOR_SEQ[i % len(COLOR_SEQ)] for i, c in enumerate(clf_list)}
    # First group by group, classifier, with/without_data2
    for (group_val, clf, is_with), subdf in df.groupby(['group', 'classifier', df['sim_name'].apply(is_with_data2)]):
        if (is_with and not show_with) or (not is_with and not show_without):
            continue
        # Sort by suffix value in ascending order
        subdf = subdf.copy()
        subdf['suffix_value'] = subdf['sim_name'].apply(lambda x: extract_suffix_value(extract_suffix(x)))
        subdf = subdf.sort_values('suffix_value')
        marker_shape = GROUP_MARKERS.get(group_val, 'circle')
        color = CLASSIFIER_COLORS.get(clf, 'black')
        sizes = subdf['sim_name'].apply(lambda x: SUFFIX_MARKER_SIZE.get(extract_suffix(x), 12)).tolist()
        legend_name = f"{group_val}-{clf}-{'with_data2' if is_with else 'without_data2'}"
        line_style = dict(width=2, dash='solid') if is_with else dict(width=2, dash='dot')
        fig.add_trace(go.Scatter(
            x=subdf['distance'],
            y=subdf['acc'],
            mode='lines+markers' if 'line' in line_check else 'markers',
            marker=dict(symbol=marker_shape, size=sizes, color=color, line=dict(width=1, color='black')),
            name=legend_name,
            text=subdf['label'],
            hoverinfo='text',
            showlegend=True,
            line=line_style if 'line' in line_check else None
        ))
    fig.update_layout(
        xaxis_title='Average Distance',
        yaxis_title='Accuracy',
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02
        ),
        margin=dict(l=60, r=250, t=60, b=60),
        hovermode='closest',
        font=dict(family='Arial', size=16)
    )
    return fig

@app.callback(
    Output('point-label', 'children'),
    [Input('main-plot', 'clickData')]
)
def display_label(clickData):
    if clickData and 'points' in clickData:
        label = clickData['points'][0]['text']
        return f'Selected: {label}'
    return ''

if __name__ == '__main__':
    app.run(debug=True, port=8050) 