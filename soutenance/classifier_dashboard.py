import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# =============================================================================
# DATA GENERATION - Circle (Real) vs Square (Simulated)
# =============================================================================
np.random.seed(42)

def generate_circle_points(n_points, center_x, center_y, radius, noise=0.12):
    """Generate points along a circle perimeter."""
    points = []
    for _ in range(n_points):
        angle = np.random.uniform(0, 2 * np.pi)
        r = radius * np.random.uniform(0.9, 1.1)
        x = center_x + r * np.cos(angle) + np.random.normal(0, noise)
        y = center_y + r * np.sin(angle) + np.random.normal(0, noise)
        points.append([x, y])
    return np.array(points)

def generate_square_points(n_points, center_x, center_y, half_size, noise=0.12):
    """Generate points along a square perimeter."""
    points = []
    points_per_side = n_points // 4
    for side in range(4):
        for _ in range(points_per_side + (1 if side < n_points % 4 else 0)):
            t = np.random.uniform(-1, 1)
            if side == 0:
                x, y = center_x + t * half_size, center_y + half_size
            elif side == 1:
                x, y = center_x + half_size, center_y + t * half_size
            elif side == 2:
                x, y = center_x + t * half_size, center_y - half_size
            else:
                x, y = center_x - half_size, center_y + t * half_size
            points.append([x + np.random.normal(0, noise), y + np.random.normal(0, noise)])
    return np.array(points)

# Parameters
CENTER_X, CENTER_Y = 5, 5
CIRCLE_RADIUS = 2.0
SQUARE_HALF = 3.5
X_MIN, X_MAX, Y_MIN, Y_MAX = 0, 10, 0, 10

# Generate data
np.random.seed(42)
real_points = generate_circle_points(50, CENTER_X, CENTER_Y, CIRCLE_RADIUS, noise=0.15)
np.random.seed(123)
sim_points = generate_square_points(50, CENTER_X, CENTER_Y, SQUARE_HALF, noise=0.15)

real_df = pd.DataFrame(real_points, columns=['x', 'y'])
real_df['true_type'] = 'Real'
sim_df = pd.DataFrame(sim_points, columns=['x', 'y'])
sim_df['true_type'] = 'Simulated'
df = pd.concat([real_df, sim_df], ignore_index=True)

# =============================================================================
# PREDICTIONS
# =============================================================================

# Linear Regression
df['lm_score'] = df['x'] + df['y']
threshold_lm = df['lm_score'].median()
df['lm_predict'] = np.where(df['lm_score'] > threshold_lm, 'Simulated', 'Real')
df['lm_correct'] = df['lm_predict'] == df['true_type']
lm_accuracy = df['lm_correct'].mean() * 100

# CNN
df['distance'] = np.sqrt((df['x'] - CENTER_X)**2 + (df['y'] - CENTER_Y)**2)
cnn_threshold = CIRCLE_RADIUS + 0.5
df['cnn_predict'] = np.where(df['distance'] <= cnn_threshold, 'Real', 'Simulated')
np.random.seed(42)
noise_mask = np.random.random(len(df)) < 0.12
df.loc[noise_mask, 'cnn_predict'] = df.loc[noise_mask, 'cnn_predict'].apply(
    lambda x: 'Simulated' if x == 'Real' else 'Real'
)
df['cnn_correct'] = df['cnn_predict'] == df['true_type']
cnn_accuracy = df['cnn_correct'].mean() * 100

# =============================================================================
# COLORS
# =============================================================================
COLORS = {
    'real': '#2E86AB',
    'simulated': '#E94F37',
    'neutral': '#8E8E93',
    'circle_fill': 'rgba(46, 134, 171, 0.18)',
    'circle_line': '#2E86AB'
}

# =============================================================================
# DASH APP
# =============================================================================
app = dash.Dash(__name__)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🔬 Classification: Linear Regression vs CNN", 
                style={'margin': '0', 'fontSize': '26px', 'fontWeight': '600'}),
        html.P("Visual comparison of simple vs complex model decision boundaries",
               style={'margin': '5px 0 0 0', 'opacity': '0.9', 'fontSize': '14px'})
    ], style={
        'textAlign': 'center', 
        'padding': '30px',
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white',
        'marginBottom': '30px'
    }),
    
    html.Div([
        # Step 1
        html.Div([
            html.Div([
                html.H2("📊 Step 1: Raw Data", style={'margin': '0 0 5px 0', 'fontSize': '18px'}),
                html.P("Before classification — Explore the hidden patterns",
                       style={'margin': '0', 'color': '#666', 'fontSize': '13px'})
            ], style={'padding': '20px 25px 15px', 'borderBottom': '1px solid #f0f0f0'}),
            
            html.Div([
                dcc.Checklist(
                    id='visibility-toggle',
                    options=[
                        {'label': ' ● Real Data (Circle)', 'value': 'real'},
                        {'label': ' ■ Simulated Data (Square)', 'value': 'sim'}
                    ],
                    value=['real', 'sim'],
                    inline=True,
                    style={'fontSize': '14px'},
                    inputStyle={'marginRight': '5px'},
                    labelStyle={'marginRight': '25px', 'cursor': 'pointer'}
                )
            ], style={
                'textAlign': 'center', 
                'padding': '15px',
                'background': '#f8f9fa',
                'margin': '0 25px 15px',
                'borderRadius': '8px'
            }),
            
            dcc.Graph(id='raw-data-viz', config={'displayModeBar': False})
        ], style={
            'background': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
            'marginBottom': '25px'
        }),
        
        # Step 2
        html.Div([
            html.Div([
                html.H2("📈 Step 2: Linear Regression", style={'margin': '0 0 5px 0', 'fontSize': '18px'}),
                html.Div([
                    html.Span("Accuracy: ", style={'color': '#666'}),
                    html.Span(f"{lm_accuracy:.0f}%", 
                             style={'fontWeight': 'bold', 'color': '#E94F37', 'fontSize': '18px'})
                ], style={'fontSize': '14px'})
            ], style={'padding': '20px 25px 15px', 'borderBottom': '1px solid #f0f0f0'}),
            
            html.Div([
                html.P("💡 Linear regression can only draw a straight line — it cannot capture circular patterns.",
                       style={'margin': '0'})
            ], style={
                'margin': '15px 25px', 'padding': '12px 15px',
                'background': '#fff3cd', 'borderRadius': '8px',
                'fontSize': '13px', 'color': '#856404'
            }),
            
            dcc.Graph(id='lm-viz', config={'displayModeBar': False}),
            
            html.Div([
                html.Span("🔵 Predicted Real"),
                html.Span("🔴 Predicted Simulated", style={'marginLeft': '15px'}),
                html.Span("⬛ Black border = Error", style={'marginLeft': '15px'})
            ], style={
                'margin': '15px 25px', 'padding': '12px 15px',
                'background': '#f8f9fa', 'borderRadius': '8px',
                'fontSize': '12px', 'color': '#666'
            })
        ], style={
            'background': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
            'marginBottom': '25px'
        }),
        
        # Step 3
        html.Div([
            html.Div([
                html.H2("🧠 Step 3: CNN", style={'margin': '0 0 5px 0', 'fontSize': '18px'}),
                html.Div([
                    html.Span("Accuracy: ", style={'color': '#666'}),
                    html.Span(f"{cnn_accuracy:.0f}%", 
                             style={'fontWeight': 'bold', 'color': '#06D6A0', 'fontSize': '18px'})
                ], style={'fontSize': '14px'})
            ], style={'padding': '20px 25px 15px', 'borderBottom': '1px solid #f0f0f0'}),
            
            html.Div([
                html.P("✨ The CNN learns complex boundaries like this circle, dramatically improving classification!",
                       style={'margin': '0'})
            ], style={
                'margin': '15px 25px', 'padding': '12px 15px',
                'background': '#d4edda', 'borderRadius': '8px',
                'fontSize': '13px', 'color': '#155724'
            }),
            
            dcc.Graph(id='cnn-viz', config={'displayModeBar': False}),
            
            html.Div([
                html.Span("🔵 Predicted Real"),
                html.Span("🔴 Predicted Simulated", style={'marginLeft': '15px'}),
                html.Span("⬛ Black border = Error", style={'marginLeft': '15px'})
            ], style={
                'margin': '15px 25px', 'padding': '12px 15px',
                'background': '#f8f9fa', 'borderRadius': '8px',
                'fontSize': '12px', 'color': '#666'
            })
        ], style={
            'background': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
            'marginBottom': '25px'
        }),
        
    ], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '0 20px'}),
    
], style={'background': '#f5f7fa', 'minHeight': '100vh', 'paddingBottom': '40px'})

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    Output('raw-data-viz', 'figure'),
    Input('visibility-toggle', 'value')
)
def update_raw_data_viz(visibility):
    fig = go.Figure()
    
    if 'real' in visibility:
        real_data = df[df['true_type'] == 'Real']
        fig.add_trace(go.Scatter(
            x=real_data['x'], y=real_data['y'], mode='markers',
            marker=dict(size=12, color=COLORS['neutral'], symbol='circle',
                       line=dict(width=1.5, color='#555')),
            name='Real (●)'
        ))
    
    if 'sim' in visibility:
        sim_data = df[df['true_type'] == 'Simulated']
        fig.add_trace(go.Scatter(
            x=sim_data['x'], y=sim_data['y'], mode='markers',
            marker=dict(size=11, color=COLORS['neutral'], symbol='square',
                       line=dict(width=1.5, color='#555')),
            name='Simulated (■)'
        ))
    
    fig.update_layout(
        title=dict(text='<b>Raw Data</b> — All gray, shapes reveal true class', font=dict(size=15)),
        xaxis=dict(range=[X_MIN, X_MAX], title='Feature 1', gridcolor='#eee', dtick=2, scaleanchor='y'),
        yaxis=dict(range=[Y_MIN, Y_MAX], title='Feature 2', gridcolor='#eee', dtick=2),
        plot_bgcolor='white', paper_bgcolor='white', height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(t=80, b=50, l=60, r=40)
    )
    return fig

@app.callback(
    Output('lm-viz', 'figure'),
    Input('lm-viz', 'id')
)
def update_lm_viz(_):
    fig = go.Figure()
    
    x_line = np.array([X_MIN, X_MAX])
    y_line = threshold_lm - x_line
    
    # Zones
    fig.add_trace(go.Scatter(
        x=[X_MIN, X_MAX, X_MAX, X_MIN],
        y=[threshold_lm - X_MIN, threshold_lm - X_MAX, Y_MAX, Y_MAX],
        fill='toself', fillcolor='rgba(233, 79, 55, 0.12)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[X_MIN, X_MAX, X_MAX, X_MIN],
        y=[threshold_lm - X_MIN, threshold_lm - X_MAX, Y_MIN, Y_MIN],
        fill='toself', fillcolor='rgba(46, 134, 171, 0.12)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        line=dict(color='#333', width=3, dash='dash'),
        name='Linear Boundary'
    ))
    
    for pred_type, color, symbol in [('Real', COLORS['real'], 'circle'), 
                                      ('Simulated', COLORS['simulated'], 'square')]:
        subset = df[df['lm_predict'] == pred_type]
        correct = subset[subset['lm_correct']]
        errors = subset[~subset['lm_correct']]
        
        if len(correct) > 0:
            fig.add_trace(go.Scatter(
                x=correct['x'], y=correct['y'], mode='markers',
                marker=dict(size=11, color=color, symbol=symbol,
                           line=dict(width=1, color='white')),
                name=f'Predicted {pred_type}'
            ))
        if len(errors) > 0:
            fig.add_trace(go.Scatter(
                x=errors['x'], y=errors['y'], mode='markers',
                marker=dict(size=11, color=color, symbol=symbol,
                           line=dict(width=3, color='black')),
                showlegend=False
            ))
    
    fig.add_annotation(x=8, y=8.5, text="Simulated<br>Zone", showarrow=False,
                      font=dict(size=12, color=COLORS['simulated']))
    fig.add_annotation(x=2, y=1.5, text="Real<br>Zone", showarrow=False,
                      font=dict(size=12, color=COLORS['real']))
    
    fig.update_layout(
        title=dict(text='<b>Linear Regression</b> — Straight line separation only', font=dict(size=15)),
        xaxis=dict(range=[X_MIN, X_MAX], title='Feature 1', gridcolor='#eee', dtick=2, scaleanchor='y'),
        yaxis=dict(range=[Y_MIN, Y_MAX], title='Feature 2', gridcolor='#eee', dtick=2),
        plot_bgcolor='white', paper_bgcolor='white', height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(t=80, b=50, l=60, r=40)
    )
    return fig

@app.callback(
    Output('cnn-viz', 'figure'),
    Input('cnn-viz', 'id')
)
def update_cnn_viz(_):
    fig = go.Figure()
    
    # Background
    fig.add_trace(go.Scatter(
        x=[X_MIN, X_MAX, X_MAX, X_MIN],
        y=[Y_MIN, Y_MIN, Y_MAX, Y_MAX],
        fill='toself', fillcolor='rgba(233, 79, 55, 0.1)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    # Circle boundary
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = CENTER_X + cnn_threshold * np.cos(theta)
    y_circle = CENTER_Y + cnn_threshold * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        fill='toself', fillcolor=COLORS['circle_fill'],
        line=dict(color=COLORS['circle_line'], width=3),
        name='Real Zone (CNN)'
    ))
    
    for pred_type, color, symbol in [('Real', COLORS['real'], 'circle'), 
                                      ('Simulated', COLORS['simulated'], 'square')]:
        subset = df[df['cnn_predict'] == pred_type]
        correct = subset[subset['cnn_correct']]
        errors = subset[~subset['cnn_correct']]
        
        if len(correct) > 0:
            fig.add_trace(go.Scatter(
                x=correct['x'], y=correct['y'], mode='markers',
                marker=dict(size=11, color=color, symbol=symbol,
                           line=dict(width=1, color='white')),
                name=f'Predicted {pred_type}'
            ))
        if len(errors) > 0:
            fig.add_trace(go.Scatter(
                x=errors['x'], y=errors['y'], mode='markers',
                marker=dict(size=11, color=color, symbol=symbol,
                           line=dict(width=3, color='black')),
                showlegend=False
            ))
    
    fig.add_annotation(x=1.3, y=8.7, text="Simulated<br>Zone", showarrow=False,
                      font=dict(size=12, color=COLORS['simulated']))
    fig.add_annotation(x=CENTER_X, y=CENTER_Y, text="Real<br>Zone", showarrow=False,
                      font=dict(size=12, color=COLORS['real']))
    
    fig.update_layout(
        title=dict(text='<b>CNN</b> — Learned circular boundary', font=dict(size=15)),
        xaxis=dict(range=[X_MIN, X_MAX], title='Feature 1', gridcolor='#eee', dtick=2, scaleanchor='y'),
        yaxis=dict(range=[Y_MIN, Y_MAX], title='Feature 2', gridcolor='#eee', dtick=2),
        plot_bgcolor='white', paper_bgcolor='white', height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(t=80, b=50, l=60, r=40)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)