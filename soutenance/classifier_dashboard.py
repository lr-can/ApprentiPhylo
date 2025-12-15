"""
Dashboard interactif pour visualiser et expliquer les modèles LM vs CNN
sur des données Réel/Simulé.

Usage:
    python classifier_dashboard.py

Le dashboard sera accessible sur http://localhost:8050
"""

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# Génération des données synthétiques
# =============================================================================

np.random.seed(42)

def generate_dataset(n_per_class=100):
    """
    Génère un dataset où les classes Réel/Simulé se chevauchent
    de manière à ce que ~50% de chaque classe soit de chaque côté
    de la frontière linéaire.
    """
    data = []
    
    for true_label, label_name in [(0, 'Réel'), (1, 'Simulé')]:
        n_half = n_per_class // 2
        
        # Moitié au-dessus de y = x
        x_above = np.random.uniform(1, 9, n_half)
        y_above = x_above + np.random.uniform(0.5, 3, n_half)
        
        # Moitié en-dessous de y = x
        x_below = np.random.uniform(1, 9, n_half)
        y_below = x_below - np.random.uniform(0.5, 3, n_half)
        
        # Patterns subtils différents selon la classe (pour que CNN performe mieux)
        if label_name == 'Réel':
            x_above += np.random.normal(0, 0.5, n_half)
            y_above += np.random.normal(0, 0.8, n_half)
            x_below += np.random.normal(0, 0.5, n_half)
            y_below += np.random.normal(0, 0.8, n_half)
        else:
            x_above += np.random.normal(0.3, 0.3, n_half)
            y_above += np.random.normal(-0.2, 0.4, n_half)
            x_below += np.random.normal(0.3, 0.3, n_half)
            y_below += np.random.normal(-0.2, 0.4, n_half)
        
        x_all = np.concatenate([x_above, x_below])
        y_all = np.concatenate([y_above, y_below])
        
        for x, y in zip(x_all, y_all):
            data.append({
                'feature_1': x,
                'feature_2': y,
                'true_type': label_name,
                'true_type_numeric': true_label
            })
    
    return pd.DataFrame(data)

# Génération
df = generate_dataset(n_per_class=100)

# Prédictions LM (frontière y = x)
df['residual_lm'] = df['feature_2'] - df['feature_1']
df['lm_predict'] = np.where(df['residual_lm'] > 0, 'Simulé', 'Réel')
df['lm_correct'] = (df['lm_predict'] == df['true_type']).map({True: 'Correct', False: 'Erreur'})

# Simulation prédictions CNN (meilleure accuracy)
def simulate_cnn(df):
    predictions = []
    for _, row in df.iterrows():
        x, y = row['feature_1'], row['feature_2']
        true_label = row['true_type']
        dist = np.sqrt((x - 5)**2 + (y - 5)**2)
        
        if true_label == 'Simulé':
            prob_correct = 0.7 + 0.1 * (1 - dist/8)
        else:
            prob_correct = 0.7 + 0.1 * (dist/8)
        
        prob_correct = np.clip(prob_correct, 0.6, 0.9)
        
        if np.random.random() < prob_correct:
            predictions.append(true_label)
        else:
            predictions.append('Simulé' if true_label == 'Réel' else 'Réel')
    return predictions

np.random.seed(123)  # Pour CNN reproductible
df['cnn_predict'] = simulate_cnn(df)
df['cnn_correct'] = (df['cnn_predict'] == df['true_type']).map({True: 'Correct', False: 'Erreur'})

# Colonnes combinées
df['lm_status'] = df['true_type'] + ' - ' + df['lm_correct']
df['cnn_status'] = df['true_type'] + ' - ' + df['cnn_correct']

# Calcul des métriques
lm_accuracy = (df['lm_predict'] == df['true_type']).mean()
cnn_accuracy = (df['cnn_predict'] == df['true_type']).mean()

# =============================================================================
# Application Dash
# =============================================================================

app = Dash(__name__)

# Couleurs
COLORS = {
    'Réel': '#3498db',
    'Simulé': '#e74c3c',
    'Correct': '#2ecc71',
    'Erreur': '#e74c3c',
    'Réel - Correct': '#2ecc71',
    'Réel - Erreur': '#c0392b',
    'Simulé - Correct': '#27ae60',
    'Simulé - Erreur': '#d35400'
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🔬 Dashboard : Comparaison LM vs CNN", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
        html.P("Visualisation interactive pour comprendre les différences entre les classifieurs",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Métriques en haut
    html.Div([
        html.Div([
            html.H3(f"{lm_accuracy:.1%}", style={'margin': '0', 'color': '#3498db', 'fontSize': '2em'}),
            html.P("Accuracy LM", style={'margin': '0', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'flex': '1'}),
        html.Div([
            html.H3(f"{cnn_accuracy:.1%}", style={'margin': '0', 'color': '#e74c3c', 'fontSize': '2em'}),
            html.P("Accuracy CNN", style={'margin': '0', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'flex': '1'}),
        html.Div([
            html.H3(f"{len(df)}", style={'margin': '0', 'color': '#2ecc71', 'fontSize': '2em'}),
            html.P("Échantillons", style={'margin': '0', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'flex': '1'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px',
              'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px',
              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Contrôles
    html.Div([
        html.Div([
            html.Label("🎨 Coloration des points :", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='color-by',
                options=[
                    {'label': '📌 Type réel (Réel/Simulé)', 'value': 'true_type'},
                    {'label': '🔵 Prédiction LM', 'value': 'lm_predict'},
                    {'label': '🔴 Prédiction CNN', 'value': 'cnn_predict'},
                    {'label': '✅ Résultat LM (Correct/Erreur)', 'value': 'lm_correct'},
                    {'label': '✅ Résultat CNN (Correct/Erreur)', 'value': 'cnn_correct'},
                    {'label': '📊 Status détaillé LM', 'value': 'lm_status'},
                    {'label': '📊 Status détaillé CNN', 'value': 'cnn_status'},
                ],
                value='true_type',
                style={'width': '100%'}
            )
        ], style={'flex': '1', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("🔷 Forme des points :", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='symbol-by',
                options=[
                    {'label': '📌 Type réel (Réel/Simulé)', 'value': 'true_type'},
                    {'label': '🔵 Prédiction LM', 'value': 'lm_predict'},
                    {'label': '🔴 Prédiction CNN', 'value': 'cnn_predict'},
                    {'label': '⭕ Aucune distinction', 'value': 'none'},
                ],
                value='true_type',
                style={'width': '100%'}
            )
        ], style={'flex': '1', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("📏 Options d'affichage :", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Checklist(
                id='display-options',
                options=[
                    {'label': ' Régression linéaire (y = x)', 'value': 'show_lm_line'},
                    {'label': ' Zone de décision LM', 'value': 'show_lm_zones'},
                    {'label': ' Annotations erreurs', 'value': 'show_errors'},
                ],
                value=['show_lm_line'],
                style={'display': 'flex', 'flexDirection': 'column', 'gap': '5px'}
            )
        ], style={'flex': '1'}),
    ], style={'display': 'flex', 'marginBottom': '20px', 'backgroundColor': 'white',
              'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Graphique principal
    html.Div([
        dcc.Graph(id='main-scatter', style={'height': '500px'})
    ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px',
              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
    
    # Graphiques de comparaison
    html.Div([
        html.Div([
            dcc.Graph(id='confusion-lm')
        ], style={'flex': '1', 'marginRight': '10px'}),
        html.Div([
            dcc.Graph(id='confusion-cnn')
        ], style={'flex': '1', 'marginLeft': '10px'}),
    ], style={'display': 'flex', 'marginBottom': '20px'}),
    
    # Distribution des résidus et comparaison
    html.Div([
        html.Div([
            dcc.Graph(id='residual-dist')
        ], style={'flex': '1', 'marginRight': '10px'}),
        html.Div([
            dcc.Graph(id='accuracy-comparison')
        ], style={'flex': '1', 'marginLeft': '10px'}),
    ], style={'display': 'flex', 'marginBottom': '20px'}),
    
    # Tableau récapitulatif
    html.Div([
        html.H3("📋 Aperçu des données", style={'color': '#2c3e50', 'marginBottom': '10px'}),
        html.Div(id='data-table')
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto',
          'padding': '20px', 'backgroundColor': '#f5f6fa'})

# =============================================================================
# Callbacks
# =============================================================================

@callback(
    Output('main-scatter', 'figure'),
    Input('color-by', 'value'),
    Input('symbol-by', 'value'),
    Input('display-options', 'value')
)
def update_main_scatter(color_by, symbol_by, display_options):
    display_options = display_options or []
    
    # Mapping des couleurs selon le choix
    if color_by in ['true_type', 'lm_predict', 'cnn_predict']:
        color_discrete_map = {'Réel': '#3498db', 'Simulé': '#e74c3c'}
    elif color_by in ['lm_correct', 'cnn_correct']:
        color_discrete_map = {'Correct': '#2ecc71', 'Erreur': '#e74c3c'}
    else:
        color_discrete_map = {
            'Réel - Correct': '#2ecc71',
            'Réel - Erreur': '#c0392b',
            'Simulé - Correct': '#27ae60',
            'Simulé - Erreur': '#d35400'
        }
    
    # Scatter plot
    fig = px.scatter(
        df,
        x='feature_1',
        y='feature_2',
        color=color_by,
        symbol=symbol_by if symbol_by != 'none' else None,
        color_discrete_map=color_discrete_map,
        symbol_map={'Réel': 'circle', 'Simulé': 'diamond'} if symbol_by != 'none' else None,
        hover_data={
            'true_type': True,
            'lm_predict': True,
            'cnn_predict': True,
            'lm_correct': True,
            'cnn_correct': True,
            'feature_1': ':.2f',
            'feature_2': ':.2f'
        },
        title='Visualisation des données et prédictions'
    )
    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    
    # Ligne de régression linéaire
    if 'show_lm_line' in display_options:
        x_line = np.linspace(df['feature_1'].min() - 1, df['feature_1'].max() + 1, 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=x_line,
            mode='lines',
            name='Frontière LM (y = x)',
            line=dict(color='black', dash='dash', width=2)
        ))
    
    # Zones de décision LM
    if 'show_lm_zones' in display_options:
        x_min, x_max = df['feature_1'].min() - 1, df['feature_1'].max() + 1
        y_min, y_max = df['feature_2'].min() - 1, df['feature_2'].max() + 1
        
        # Zone "Simulé" (au-dessus de y = x)
        fig.add_trace(go.Scatter(
            x=[x_min, x_max, x_max, x_min, x_min],
            y=[x_min, x_max, y_max, y_max, x_min],
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Zone LM → Simulé',
            hoverinfo='skip'
        ))
        
        # Zone "Réel" (en-dessous de y = x)
        fig.add_trace(go.Scatter(
            x=[x_min, x_max, x_max, x_min, x_min],
            y=[y_min, y_min, x_max, x_min, y_min],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Zone LM → Réel',
            hoverinfo='skip'
        ))
    
    # Annotations des erreurs
    if 'show_errors' in display_options:
        errors = df[(df['lm_predict'] != df['true_type']) | (df['cnn_predict'] != df['true_type'])]
        for _, row in errors.iterrows():
            annotation_text = []
            if row['lm_predict'] != row['true_type']:
                annotation_text.append('LM❌')
            if row['cnn_predict'] != row['true_type']:
                annotation_text.append('CNN❌')
            
            fig.add_annotation(
                x=row['feature_1'],
                y=row['feature_2'],
                text=' '.join(annotation_text),
                showarrow=False,
                yshift=15,
                font=dict(size=8, color='#e74c3c')
            )
    
    fig.update_layout(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        legend_title='Légende',
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

@callback(
    Output('confusion-lm', 'figure'),
    Input('color-by', 'value')  # Dummy input pour trigger
)
def update_confusion_lm(_):
    confusion = pd.crosstab(df['true_type'], df['lm_predict'])
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion.values,
        x=confusion.columns,
        y=confusion.index,
        colorscale='Blues',
        text=confusion.values,
        texttemplate='%{text}',
        textfont=dict(size=20),
        hovertemplate='Vrai: %{y}<br>Prédit: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Matrice de confusion - LM (Acc: {lm_accuracy:.1%})',
        xaxis_title='Prédit',
        yaxis_title='Vrai',
        template='plotly_white'
    )
    
    return fig

@callback(
    Output('confusion-cnn', 'figure'),
    Input('color-by', 'value')
)
def update_confusion_cnn(_):
    confusion = pd.crosstab(df['true_type'], df['cnn_predict'])
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion.values,
        x=confusion.columns,
        y=confusion.index,
        colorscale='Reds',
        text=confusion.values,
        texttemplate='%{text}',
        textfont=dict(size=20),
        hovertemplate='Vrai: %{y}<br>Prédit: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Matrice de confusion - CNN (Acc: {cnn_accuracy:.1%})',
        xaxis_title='Prédit',
        yaxis_title='Vrai',
        template='plotly_white'
    )
    
    return fig

@callback(
    Output('residual-dist', 'figure'),
    Input('color-by', 'value')
)
def update_residual_dist(_):
    fig = go.Figure()
    
    for true_type, color in [('Réel', '#3498db'), ('Simulé', '#e74c3c')]:
        subset = df[df['true_type'] == true_type]
        fig.add_trace(go.Histogram(
            x=subset['residual_lm'],
            name=true_type,
            marker_color=color,
            opacity=0.7,
            nbinsx=20
        ))
    
    fig.add_vline(x=0, line_dash='dash', line_color='black', 
                  annotation_text='Frontière LM', annotation_position='top')
    
    fig.update_layout(
        title='Distribution des résidus (y - x) par type réel',
        xaxis_title='Résidu (y - x)',
        yaxis_title='Fréquence',
        barmode='overlay',
        template='plotly_white',
        legend_title='Type réel'
    )
    
    return fig

@callback(
    Output('accuracy-comparison', 'figure'),
    Input('color-by', 'value')
)
def update_accuracy_comparison(_):
    # Calcul des métriques par classe
    metrics = []
    for model, pred_col in [('LM', 'lm_predict'), ('CNN', 'cnn_predict')]:
        for true_type in ['Réel', 'Simulé']:
            subset = df[df['true_type'] == true_type]
            acc = (subset[pred_col] == subset['true_type']).mean()
            metrics.append({
                'Modèle': model,
                'Classe': true_type,
                'Accuracy': acc
            })
    
    metrics_df = pd.DataFrame(metrics)
    
    fig = px.bar(
        metrics_df,
        x='Modèle',
        y='Accuracy',
        color='Classe',
        barmode='group',
        color_discrete_map={'Réel': '#3498db', 'Simulé': '#e74c3c'},
        text_auto='.1%'
    )
    
    fig.update_layout(
        title='Accuracy par modèle et par classe',
        yaxis_title='Accuracy',
        yaxis_tickformat='.0%',
        template='plotly_white'
    )
    
    return fig

@callback(
    Output('data-table', 'children'),
    Input('color-by', 'value')
)
def update_table(_):
    display_cols = ['feature_1', 'feature_2', 'true_type', 'lm_predict', 'lm_correct', 'cnn_predict', 'cnn_correct']
    sample = df[display_cols].head(10).round(2)
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col, style={'padding': '10px', 'backgroundColor': '#3498db', 'color': 'white'}) 
                           for col in sample.columns])),
        html.Tbody([
            html.Tr([
                html.Td(sample.iloc[i][col], style={
                    'padding': '8px',
                    'backgroundColor': '#f8f9fa' if i % 2 == 0 else 'white',
                    'color': '#2ecc71' if 'correct' in col.lower() and sample.iloc[i][col] == 'Correct' 
                            else '#e74c3c' if 'correct' in col.lower() and sample.iloc[i][col] == 'Erreur'
                            else '#2c3e50'
                }) for col in sample.columns
            ]) for i in range(len(sample))
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'textAlign': 'center'})

# =============================================================================
# Lancement
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Dashboard démarré!")
    print("📍 Ouvrez votre navigateur sur: http://localhost:8050")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8050)
