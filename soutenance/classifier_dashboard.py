"""
Interactive Dashboard for Phylogenetic Classification Pipeline
Adapted for PowerPoint presentation - Shows real data, simulated data, and classifier workflows
IMPROVED VERSION with 3D visualization and enhanced decision boundaries
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import polars as pl
from Bio import SeqIO
import sys
from typing import Optional, Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from skimage import measure
import pickle
import json

# Page configuration
st.set_page_config(
    page_title="Phylogenetic Classifiers Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .classifier-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .story-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_REAL_PATH = PROJECT_ROOT / "data" / "prot_mammals"
DEFAULT_SIM_PATH = PROJECT_ROOT / "results" / "simulations"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "classification"

# Available classifiers
CLASSIFIERS = {
    "LogisticRegressionClassifier": {
        "type": "Logistic Regression",
        "description": "Linear model based on global MSA composition",
        "data_type": "MsaCompositionData",
        "color": "#FF6B6B"
    },
    "AACnnClassifier": {
        "type": "CNN (Amino Acids)",
        "description": "Convolutional neural network for protein sequences",
        "data_type": "SiteCompositionData",
        "color": "#4ECDC4"
    },
    "DNACnnClassifier": {
        "type": "CNN (DNA)",
        "description": "Convolutional neural network for DNA sequences",
        "data_type": "SiteCompositionData",
        "color": "#45B7D1"
    },
    "DenseMsaClassifier": {
        "type": "Dense Neural Network (MSA)",
        "description": "Dense network on global MSA composition",
        "data_type": "MsaCompositionData",
        "color": "#96CEB4"
    },
    "DenseSiteClassifier": {
        "type": "Dense Neural Network (Site)",
        "description": "Dense network on per-site composition",
        "data_type": "SiteCompositionData",
        "color": "#FFEAA7"
    },
    "AttentionClassifier": {
        "type": "Transformer (Attention)",
        "description": "Transformer model with attention mechanism",
        "data_type": "SequencesData",
        "color": "#DDA0DD"
    }
}


def load_fasta_sample(file_path: Path, max_seqs: int = 5) -> List[Dict]:
    """Load a sample of sequences from a FASTA file"""
    try:
        sequences = []
        for i, record in enumerate(SeqIO.parse(file_path, "fasta")):
            if i >= max_seqs:
                break
            sequences.append({
                "id": record.id,
                "seq": str(record.seq)[:100] + "..." if len(record.seq) > 100 else str(record.seq),
                "length": len(record.seq)
            })
        return sequences
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []


def find_matching_real_sim_pairs(real_path: Path, sim_path: Path, max_pairs: int = 5) -> List[Tuple[Path, Path]]:
    """Find matching pairs of real and simulated alignments (same filename)"""
    pairs = []
    if not real_path.exists() or not sim_path.exists():
        return pairs
    
    real_files = {f.stem: f for f in real_path.glob("*.fasta")}
    sim_files = {f.stem: f for f in sim_path.glob("*.fasta")}
    
    # Find common stems
    common_stems = set(real_files.keys()) & set(sim_files.keys())
    
    for stem in list(common_stems)[:max_pairs]:
        pairs.append((real_files[stem], sim_files[stem]))
    
    return pairs


def count_fasta_files(directory: Path) -> int:
    """Count FASTA files in a directory"""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.fasta")))


def load_training_history(results_path: Path, classifier: str, run: int = 1) -> Optional[pd.DataFrame]:
    """Load training history for a classifier"""
    history_path = results_path / f"run_{run}" / classifier / "train_history.parquet"
    if history_path.exists():
        try:
            return pd.read_parquet(history_path)
        except Exception:
            return None
    return None


def load_predictions(results_path: Path, classifier: str, run: int = 1) -> Optional[pd.DataFrame]:
    """Load predictions for a classifier"""
    pred_path = results_path / f"run_{run}" / classifier / "best_preds.parquet"
    if pred_path.exists():
        try:
            return pd.read_parquet(pred_path)
        except Exception:
            return None
    
    # Try global file as well
    global_pred_path = results_path / f"run_{run}" / f"preds_run{run}.parquet"
    if global_pred_path.exists():
        try:
            df = pd.read_parquet(global_pred_path)
            if "classifier" in df.columns:
                return df[df["classifier"] == classifier]
            return df
        except Exception:
            return None
    return None


def load_roc_data(results_path: Path, classifier: str, run: int = 1) -> Optional[pd.DataFrame]:
    """Load ROC data for a classifier"""
    roc_path = results_path / f"run_{run}" / "roc_data" / f"{classifier}_roc.csv"
    if roc_path.exists():
        try:
            return pd.read_csv(roc_path)
        except Exception:
            return None
    return None


def load_classifier_model(results_path: Path, classifier: str, run: int = 1):
    """Load a trained classifier model if available"""
    # For LogisticRegression, check for pickle or JSON summary
    if classifier == "LogisticRegressionClassifier":
        summary_path = results_path / f"run_{run}" / classifier / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def extract_features_from_data(real_path: Path, sim_path: Path, classifier: str, max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from alignments for visualization.
    For LogisticRegression, uses MSA composition.
    For others, uses predictions as features or simulated embeddings.
    """
    try:
        # Import classifier modules
        sys.path.insert(0, str(PROJECT_ROOT / "backup" / "simulations-classifiers" / "src"))
        try:
            from classifiers.data.sources import FastaSource  # type: ignore
            from classifiers.data.data import Data  # type: ignore
            from classifiers.data import tokenizers  # type: ignore
        except ImportError:
            # If imports fail, return dummy data
            raise ImportError("Classifier modules not available")
        
        # Load data
        real_source = FastaSource(real_path)
        sim_source = FastaSource(sim_path)
        
        # Use AA tokenizer by default
        tokenizer = tokenizers.AA_TOKENIZER
        
        data = Data(
            source_real=real_source,
            source_simulated=sim_source,
            tokenizer=tokenizer
        )
        
        # Get labels
        labels_dict = data.labels
        filenames = list(labels_dict.keys())
        labels = np.array([labels_dict[f] for f in filenames])
        
        # Limit samples for performance
        if len(filenames) > max_samples:
            indices = np.random.choice(len(filenames), max_samples, replace=False)
            filenames = [filenames[i] for i in indices]
            labels = labels[indices]
        
        # Extract features based on classifier type
        if classifier == "LogisticRegressionClassifier":
            # Use MSA composition features
            from classifiers.data import preprocessing_fn  # type: ignore
            msa_features = preprocessing_fn.msa_composition_preprocessing(
                data.aligns,
                data.n_tokens
            )
            features = np.array([msa_features[f] for f in filenames if f in msa_features])
            # Filter labels to match
            valid_indices = [i for i, f in enumerate(filenames) if f in msa_features]
            labels = labels[valid_indices]
            filenames = [filenames[i] for i in valid_indices]
        else:
            # For other classifiers, use predictions as features
            # Or create a simplified feature representation
            from classifiers.data import preprocessing_fn  # type: ignore
            site_features = preprocessing_fn.site_composition_preprocessing(
                data.aligns,
                data.n_tokens
            )
            # Average over sites to get a fixed-size feature vector
            feature_list = []
            valid_filenames = []
            valid_labels = []
            for fname in filenames:
                if fname in site_features:
                    feat = site_features[fname]
                    # Average composition across sites
                    avg_feat = feat.mean(axis=0)
                    feature_list.append(avg_feat)
                    valid_filenames.append(fname)
                    valid_labels.append(labels_dict[fname])
            features = np.array(feature_list)
            labels = np.array(valid_labels)
            filenames = valid_filenames
        
        return features, labels, filenames
    
    except Exception as e:
        st.warning(f"Could not extract features: {e}")
        # Return dummy data for demonstration
        n_samples = 200
        n_features = 20
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 2, n_samples)
        filenames = [f"sample_{i}" for i in range(n_samples)]
        return features, labels, filenames


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """Compute PCA projection of features to 2D or 3D"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=n_components)
    pca_projection = pca.fit_transform(features_scaled)
    
    return pca_projection, pca, scaler


def compute_tsne_projection(features: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> Tuple[np.ndarray, TSNE, np.ndarray]:
    """Compute t-SNE projection of features to 2D or 3D"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Limit samples for t-SNE (it's computationally expensive)
    max_samples = 1000
    if len(features_scaled) > max_samples:
        indices = np.random.choice(len(features_scaled), max_samples, replace=False)
        features_scaled = features_scaled[indices]
        sample_indices = indices
    else:
        sample_indices = np.arange(len(features_scaled))
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, max_iter=1000)
    tsne_projection = tsne.fit_transform(features_scaled)
    
    return tsne_projection, tsne, sample_indices


def plot_3d_separation(
    features_3d: np.ndarray,
    labels: np.ndarray,
    classifier_name: str,
    original_features: Optional[np.ndarray] = None,
    pca: Optional[PCA] = None,
    scaler: Optional[StandardScaler] = None,
    predictions: Optional[np.ndarray] = None,
    projection_type: str = "PCA",
    filenames: Optional[List[str]] = None
) -> Tuple[go.Figure, Dict[str, float]]:
    """
    Plot 3D projection with decision boundaries
    """
    fig = go.Figure()
    
    color = CLASSIFIERS.get(classifier_name, {}).get("color", "#1f77b4")
    
    # Separate real and simulated
    real_mask = labels == 1
    sim_mask = labels == 0
    
    # Calculate performance metrics
    performance_metrics = {}
    
    # If we have predictions, calculate performance
    if predictions is not None:
        pred_classes = (predictions >= 0.5).astype(int)
        correct = (pred_classes == labels)
        accuracy = correct.mean()
        performance_metrics['accuracy'] = accuracy
        
        # Calculate errors
        errors = ~correct
        performance_metrics['n_errors'] = errors.sum()
        performance_metrics['error_rate'] = errors.mean()
        
        # Separate correct and incorrect predictions
        correct_real = real_mask & correct
        correct_sim = sim_mask & correct
        error_real = real_mask & errors
        error_sim = sim_mask & errors
    else:
        correct_real = real_mask
        correct_sim = sim_mask
        error_real = np.zeros_like(real_mask, dtype=bool)
        error_sim = np.zeros_like(sim_mask, dtype=bool)
        performance_metrics = {'accuracy': None, 'n_errors': 0, 'error_rate': 0.0}
    
    # FIRST: Add decision surface if predictions available
    if predictions is not None and len(predictions) == len(features_3d):
        try:
            # Train a classifier in 3D space for visualization
            lr_3d = LogisticRegression(max_iter=1000, random_state=42)
            lr_3d.fit(features_3d, labels)
            
            # Create a 3D grid for the decision surface
            x_min, x_max = features_3d[:, 0].min() - 1, features_3d[:, 0].max() + 1
            y_min, y_max = features_3d[:, 1].min() - 1, features_3d[:, 1].max() + 1
            z_min, z_max = features_3d[:, 2].min() - 1, features_3d[:, 2].max() + 1
            
            # Create a mesh grid for the decision surface
            # We'll create an isosurface at P=0.5
            resolution = 30
            xx, yy, zz = np.meshgrid(
                np.linspace(x_min, x_max, resolution),
                np.linspace(y_min, y_max, resolution),
                np.linspace(z_min, z_max, resolution)
            )
            
            # Predict probabilities on the grid
            grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            probs = lr_3d.predict_proba(grid_points)[:, 1]
            probs = probs.reshape(xx.shape)
            
            # Add isosurface at P=0.5 (decision boundary)
            fig.add_trace(go.Isosurface(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=probs.flatten(),
                isomin=0.45,
                isomax=0.55,
                surface_count=1,
                colorscale=[[0, 'rgba(255, 215, 0, 0.4)'], [1, 'rgba(255, 215, 0, 0.4)']],
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                name='Decision Surface (P=0.5)',
                hoverinfo='skip',
                legendrank=1000
            ))
            
            # Add probability clouds for better visualization
            # Real zone cloud (P > 0.7)
            fig.add_trace(go.Isosurface(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=probs.flatten(),
                isomin=0.7,
                isomax=1.0,
                surface_count=2,
                colorscale=[[0, 'rgba(46, 134, 171, 0.15)'], [1, 'rgba(46, 134, 171, 0.25)']],
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                name='Real Zone (P>0.7)',
                hoverinfo='skip',
                legendrank=999
            ))
            
            # Simulated zone cloud (P < 0.3)
            fig.add_trace(go.Isosurface(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=probs.flatten(),
                isomin=0.0,
                isomax=0.3,
                surface_count=2,
                colorscale=[[0, 'rgba(162, 59, 114, 0.25)'], [1, 'rgba(162, 59, 114, 0.15)']],
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                name='Simulated Zone (P<0.3)',
                hoverinfo='skip',
                legendrank=998
            ))
            
        except Exception as e:
            st.warning(f"Could not create 3D decision surface: {e}")
    
    # SECOND: Plot data points (foreground)
    # Plot correctly classified real alignments
    if correct_real.sum() > 0:
        correct_real_indices = np.where(correct_real)[0]
        marker_size = 6
        marker_opacity = 0.8
        marker_color = '#2E86AB'
        
        if predictions is not None:
            probs = predictions[correct_real_indices]
            marker_size = 4 + (probs * 6)
            marker_opacity = 0.6 + (probs * 0.4)
        
        fig.add_trace(go.Scatter3d(
            x=features_3d[correct_real_indices, 0],
            y=features_3d[correct_real_indices, 1],
            z=features_3d[correct_real_indices, 2],
            mode='markers',
            name='Real (Correct)',
            marker=dict(
                color=marker_color,
                size=marker_size if isinstance(marker_size, (int, float)) else 6,
                opacity=marker_opacity if isinstance(marker_opacity, (int, float)) else 0.8,
                line=dict(width=1, color='darkblue')
            ),
            text=[f"Real (P={predictions[i]:.3f})" if predictions is not None else f"Real {i}" 
                  for i in correct_real_indices],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Plot correctly classified simulated alignments
    if correct_sim.sum() > 0:
        correct_sim_indices = np.where(correct_sim)[0]
        marker_size = 6
        marker_opacity = 0.8
        marker_color = '#A23B72'
        
        if predictions is not None:
            probs = 1 - predictions[correct_sim_indices]
            marker_size = 4 + (probs * 6)
            marker_opacity = 0.6 + (probs * 0.4)
        
        fig.add_trace(go.Scatter3d(
            x=features_3d[correct_sim_indices, 0],
            y=features_3d[correct_sim_indices, 1],
            z=features_3d[correct_sim_indices, 2],
            mode='markers',
            name='Simulated (Correct)',
            marker=dict(
                color=marker_color,
                size=marker_size if isinstance(marker_size, (int, float)) else 6,
                opacity=marker_opacity if isinstance(marker_opacity, (int, float)) else 0.8,
                line=dict(width=1, color='darkred')
            ),
            text=[f"Simulated (P={1-predictions[i]:.3f})" if predictions is not None else f"Simulated {i}"
                  for i in correct_sim_indices],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Plot misclassified real alignments
    if error_real.sum() > 0:
        error_real_indices = np.where(error_real)[0]
        fig.add_trace(go.Scatter3d(
            x=features_3d[error_real_indices, 0],
            y=features_3d[error_real_indices, 1],
            z=features_3d[error_real_indices, 2],
            mode='markers',
            name='Real (Misclassified)',
            marker=dict(
                symbol='x',
                size=10,
                color='red',
                opacity=1.0,
                line=dict(width=2, color='darkred')
            ),
            text=[f"ERROR: Real labeled as Sim (P={predictions[i]:.3f})" if predictions is not None else f"ERROR: Real {i}"
                  for i in error_real_indices],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Plot misclassified simulated alignments
    if error_sim.sum() > 0:
        error_sim_indices = np.where(error_sim)[0]
        fig.add_trace(go.Scatter3d(
            x=features_3d[error_sim_indices, 0],
            y=features_3d[error_sim_indices, 1],
            z=features_3d[error_sim_indices, 2],
            mode='markers',
            name='Simulated (Misclassified)',
            marker=dict(
                symbol='x',
                size=10,
                color='orange',
                opacity=1.0,
                line=dict(width=2, color='darkorange')
            ),
            text=[f"ERROR: Sim labeled as Real (P={predictions[i]:.3f})" if predictions is not None else f"ERROR: Simulated {i}"
                  for i in error_sim_indices],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # If no predictions, plot all points
    if predictions is None:
        if real_mask.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=features_3d[real_mask, 0],
                y=features_3d[real_mask, 1],
                z=features_3d[real_mask, 2],
                mode='markers',
                name='Real',
                marker=dict(
                    color='#2E86AB',
                    size=6,
                    opacity=0.8,
                    line=dict(width=1, color='darkblue')
                ),
                text=[f"Real {i}" for i in range(real_mask.sum())],
                hovertemplate='Real Alignment<br>%{text}<extra></extra>'
            ))
        
        if sim_mask.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=features_3d[sim_mask, 0],
                y=features_3d[sim_mask, 1],
                z=features_3d[sim_mask, 2],
                mode='markers',
                name='Simulated',
                marker=dict(
                    color='#A23B72',
                    size=6,
                    opacity=0.8,
                    line=dict(width=1, color='darkred')
                ),
                text=[f"Simulated {i}" for i in range(sim_mask.sum())],
                hovertemplate='Simulated Alignment<br>%{text}<extra></extra>'
            ))
    
    # Set axis labels
    if projection_type == "t-SNE":
        x_label = "t-SNE Dim 1"
        y_label = "t-SNE Dim 2"
        z_label = "t-SNE Dim 3"
    else:
        x_label = "PC1"
        y_label = "PC2"
        z_label = "PC3"
    
    # Add title with performance metrics
    title = f"3D Projection - {classifier_name}"
    if performance_metrics.get('accuracy') is not None:
        title += f"<br><sub>Accuracy: {performance_metrics['accuracy']:.1%} | Errors: {performance_metrics['n_errors']}/{len(labels)}</sub>"
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=800,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig, performance_metrics


def plot_2d_separation(
    features_2d: np.ndarray,
    labels: np.ndarray,
    classifier_name: str,
    original_features: Optional[np.ndarray] = None,
    pca: Optional[PCA] = None,
    scaler: Optional[StandardScaler] = None,
    predictions: Optional[np.ndarray] = None,
    model: Optional[object] = None,
    projection_type: str = "PCA",
    filenames: Optional[List[str]] = None
) -> Tuple[go.Figure, Dict[str, float]]:
    """
    Plot 2D projection with decision boundaries and model performance
    Shows predictions, probabilities, and classification errors
    Returns figure and performance metrics
    """
    fig = go.Figure()
    
    color = CLASSIFIERS.get(classifier_name, {}).get("color", "#1f77b4")
    
    # Separate real and simulated
    real_mask = labels == 1
    sim_mask = labels == 0
    
    # Calculate performance metrics
    performance_metrics = {}
    
    # If we have predictions, calculate performance
    if predictions is not None:
        pred_classes = (predictions >= 0.5).astype(int)
        correct = (pred_classes == labels)
        accuracy = correct.mean()
        performance_metrics['accuracy'] = accuracy
        
        # Calculate errors
        errors = ~correct
        performance_metrics['n_errors'] = errors.sum()
        performance_metrics['error_rate'] = errors.mean()
        
        # Separate correct and incorrect predictions
        correct_real = real_mask & correct
        correct_sim = sim_mask & correct
        error_real = real_mask & errors  # Real labeled as simulated
        error_sim = sim_mask & errors     # Simulated labeled as real
    else:
        correct_real = real_mask
        correct_sim = sim_mask
        error_real = np.zeros_like(real_mask, dtype=bool)
        error_sim = np.zeros_like(sim_mask, dtype=bool)
        performance_metrics = {'accuracy': None, 'n_errors': 0, 'error_rate': 0.0}
    
    # FIRST: Add decision boundary zones (background)
    if predictions is not None and len(predictions) == len(features_2d):
        try:
            # Create a grid in 2D space
            x_min, x_max = features_2d[:, 0].min() - 0.5, features_2d[:, 0].max() + 0.5
            y_min, y_max = features_2d[:, 1].min() - 0.5, features_2d[:, 1].max() + 0.5
            
            # Create fine grid
            grid_resolution = 150
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, grid_resolution),
                np.linspace(y_min, y_max, grid_resolution)
            )
            
            # Train a 2D classifier for visualization
            try:
                lr_2d = LogisticRegression(max_iter=1000, random_state=42)
                lr_2d.fit(features_2d, labels)
                
                # Get decision boundary probabilities
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                Z = lr_2d.predict_proba(grid_points)[:, 1]
                Z = Z.reshape(xx.shape)
                
                # Create VERY visible classification zones
                # Use a single contour with clear color separation at P=0.5
                # Colors: Purple for Simulated, Blue for Real
                fig.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, grid_resolution),
                    y=np.linspace(y_min, y_max, grid_resolution),
                    z=Z,
                    colorscale=[
                        [0, 'rgba(162, 59, 114, 0.7)'],      # Simulated zone - purple
                        [0.45, 'rgba(162, 59, 114, 0.6)'],
                        [0.5, 'rgba(150, 150, 150, 0.4)'],   # Boundary area
                        [0.55, 'rgba(46, 134, 171, 0.6)'],
                        [1, 'rgba(46, 134, 171, 0.7)']       # Real zone - blue
                    ],
                    opacity=1.0,
                    showscale=True,
                    colorbar=dict(
                        title="P(Real)", 
                        x=1.02, 
                        len=0.5, 
                        y=0.75,
                        tickvals=[0, 0.5, 1.0],
                        ticktext=['0.0<br>Simulated', '0.5<br>Boundary', '1.0<br>Real']
                    ),
                    contours=dict(
                        start=0,
                        end=1,
                        size=0.1,
                        showlines=False,
                        coloring='fill'
                    ),
                    name='Classification Zones',
                    legendrank=1000,
                    hoverinfo='skip'
                ))
                
                # Add prominent decision boundary line at P=0.5
                # Try using skimage first, fallback to manual extraction
                try:
                    from skimage import measure
                    contour_0_5 = measure.find_contours(Z, 0.5)
                    for contour in contour_0_5:
                        # Convert contour indices to actual coordinates
                        contour_x = np.interp(contour[:, 1], np.arange(grid_resolution), np.linspace(x_min, x_max, grid_resolution))
                        contour_y = np.interp(contour[:, 0], np.arange(grid_resolution), np.linspace(y_min, y_max, grid_resolution))
                        
                        fig.add_trace(go.Scatter(
                            x=contour_x,
                            y=contour_y,
                            mode='lines',
                            line=dict(color='black', width=5),
                            name='Decision Boundary (P=0.5)',
                            showlegend=True,
                            legendrank=999,
                            hoverinfo='skip',
                            fill='toself' if len(contour) > 2 else None
                        ))
                except ImportError:
                    # Fallback: extract boundary points manually
                    boundary_x = []
                    boundary_y = []
                    x_coords = np.linspace(x_min, x_max, grid_resolution)
                    y_coords = np.linspace(y_min, y_max, grid_resolution)
                    
                    threshold = 0.03
                    for i in range(grid_resolution - 1):
                        for j in range(grid_resolution - 1):
                            z00, z01, z10, z11 = Z[i, j], Z[i, j+1], Z[i+1, j], Z[i+1, j+1]
                            # Check if 0.5 is crossed in this cell
                            if (z00 <= 0.5 <= z01) or (z01 <= 0.5 <= z00) or \
                               (z00 <= 0.5 <= z10) or (z10 <= 0.5 <= z00) or \
                               (z11 <= 0.5 <= z01) or (z01 <= 0.5 <= z11) or \
                               (z11 <= 0.5 <= z10) or (z10 <= 0.5 <= z11):
                                # Add all four corners if close to boundary
                                if abs(z00 - 0.5) < threshold:
                                    boundary_x.append(x_coords[j])
                                    boundary_y.append(y_coords[i])
                                if abs(z01 - 0.5) < threshold:
                                    boundary_x.append(x_coords[j+1])
                                    boundary_y.append(y_coords[i])
                                if abs(z10 - 0.5) < threshold:
                                    boundary_x.append(x_coords[j])
                                    boundary_y.append(y_coords[i+1])
                                if abs(z11 - 0.5) < threshold:
                                    boundary_x.append(x_coords[j+1])
                                    boundary_y.append(y_coords[i+1])
                    
                    if boundary_x:
                        # Sort and connect points
                        boundary_points = list(zip(boundary_x, boundary_y))
                        boundary_points.sort(key=lambda p: (p[0], p[1]))
                        sorted_x = [p[0] for p in boundary_points]
                        sorted_y = [p[1] for p in boundary_points]
                        
                        fig.add_trace(go.Scatter(
                            x=sorted_x,
                            y=sorted_y,
                            mode='lines',
                            line=dict(color='black', width=5),
                            name='Decision Boundary (P=0.5)',
                            showlegend=True,
                            legendrank=999,
                            hoverinfo='skip'
                        ))
                
                # Add clear zone labels - positioned in the center of each zone
                # Find the center of each classification zone based on predictions
                if predictions is not None:
                    real_pred_mask = predictions > 0.5
                    sim_pred_mask = predictions <= 0.5
                    
                    if real_pred_mask.sum() > 0:
                        real_center_x = features_2d[real_pred_mask, 0].mean()
                        real_center_y = features_2d[real_pred_mask, 1].mean()
                    else:
                        real_center_x = (x_max + x_min) / 2 + (x_max - x_min) * 0.2
                        real_center_y = (y_max + y_min) / 2
                    
                    if sim_pred_mask.sum() > 0:
                        sim_center_x = features_2d[sim_pred_mask, 0].mean()
                        sim_center_y = features_2d[sim_pred_mask, 1].mean()
                    else:
                        sim_center_x = (x_max + x_min) / 2 - (x_max - x_min) * 0.2
                        sim_center_y = (y_max + y_min) / 2
                else:
                    # Fallback to label-based positioning
                    real_center_x = features_2d[labels == 1, 0].mean() if (labels == 1).sum() > 0 else (x_max + x_min) / 2 + (x_max - x_min) * 0.2
                    real_center_y = features_2d[labels == 1, 1].mean() if (labels == 1).sum() > 0 else (y_max + y_min) / 2
                    sim_center_x = features_2d[labels == 0, 0].mean() if (labels == 0).sum() > 0 else (x_max + x_min) / 2 - (x_max - x_min) * 0.2
                    sim_center_y = features_2d[labels == 0, 1].mean() if (labels == 0).sum() > 0 else (y_max + y_min) / 2
                
                # Add REAL ZONE label
                fig.add_annotation(
                    x=real_center_x, y=real_center_y,
                    text="<b>REAL ZONE</b><br>Model predicts: Real",
                    showarrow=False,
                    font=dict(size=18, color="white", family="Arial Black"),
                    bgcolor="rgba(46, 134, 171, 0.85)",
                    bordercolor="rgba(46, 134, 171, 1)",
                    borderwidth=4,
                    borderpad=10,
                    xref="x", yref="y"
                )
                
                # Add SIMULATED ZONE label
                fig.add_annotation(
                    x=sim_center_x, y=sim_center_y,
                    text="<b>SIMULATED ZONE</b><br>Model predicts: Simulated",
                    showarrow=False,
                    font=dict(size=18, color="white", family="Arial Black"),
                    bgcolor="rgba(162, 59, 114, 0.85)",
                    bordercolor="rgba(162, 59, 114, 1)",
                    borderwidth=4,
                    borderpad=10,
                    xref="x", yref="y"
                )
                
            except Exception as e:
                st.warning(f"Could not create decision boundary: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        except Exception as e:
            st.warning(f"Could not plot decision zones: {e}")
    
    # SECOND: Plot data points (foreground) - simplified: just Real and Simulated
    # Plot all real alignments
    if real_mask.sum() > 0:
        real_indices = np.where(real_mask)[0]
        fig.add_trace(go.Scatter(
            x=features_2d[real_indices, 0],
            y=features_2d[real_indices, 1],
            mode='markers',
            name='Real Alignments',
            marker=dict(
                color='#2E86AB',
                size=12,
                opacity=0.9,
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            text=[f"Real (P={predictions[i]:.3f})" if predictions is not None else f"Real {i}" 
                  for i in real_indices],
            hovertemplate='%{text}<extra></extra>',
            legendrank=1
        ))
    
    # Plot all simulated alignments
    if sim_mask.sum() > 0:
        sim_indices = np.where(sim_mask)[0]
        fig.add_trace(go.Scatter(
            x=features_2d[sim_indices, 0],
            y=features_2d[sim_indices, 1],
            mode='markers',
            name='Simulated Alignments',
            marker=dict(
                color='#A23B72',
                size=12,
                opacity=0.9,
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            text=[f"Simulated (P={1-predictions[i]:.3f})" if predictions is not None else f"Simulated {i}"
                  for i in sim_indices],
            hovertemplate='%{text}<extra></extra>',
            legendrank=2
        ))
    
    # Plot errors - highlight misclassifications
    if predictions is not None:
        # Real alignments misclassified as simulated
        if error_real.sum() > 0:
            error_real_indices = np.where(error_real)[0]
            fig.add_trace(go.Scatter(
                x=features_2d[error_real_indices, 0],
                y=features_2d[error_real_indices, 1],
                mode='markers',
                name='Error: Real → Simulated',
                marker=dict(
                    symbol='x',
                    size=25,
                    color='red',
                    opacity=1.0,
                    line=dict(width=4, color='darkred')
                ),
                text=[f"ERROR: Real misclassified as Sim (P={predictions[i]:.3f})"
                      for i in error_real_indices],
                hovertemplate='%{text}<extra></extra>',
                legendrank=3
            ))
        
        # Simulated alignments misclassified as real
        if error_sim.sum() > 0:
            error_sim_indices = np.where(error_sim)[0]
            fig.add_trace(go.Scatter(
                x=features_2d[error_sim_indices, 0],
                y=features_2d[error_sim_indices, 1],
                mode='markers',
                name='Error: Simulated → Real',
                marker=dict(
                    symbol='x',
                    size=25,
                    color='orange',
                    opacity=1.0,
                    line=dict(width=4, color='darkorange')
                ),
                text=[f"ERROR: Simulated misclassified as Real (P={predictions[i]:.3f})"
                      for i in error_sim_indices],
                hovertemplate='%{text}<extra></extra>',
                legendrank=4
            ))
    
    # Set axis labels
    if projection_type == "t-SNE":
        x_label = "t-SNE Dimension 1"
        y_label = "t-SNE Dimension 2"
    else:
        x_label = "First Principal Component (PC1)"
        y_label = "Second Principal Component (PC2)"
    
    # Add title with performance metrics
    title = f"Classification Zones in 2D Space - {classifier_name}"
    if performance_metrics.get('accuracy') is not None:
        title += f"<br><sub>Accuracy: {performance_metrics['accuracy']:.1%} | Errors: {performance_metrics['n_errors']}/{len(labels)}</sub>"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=700,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig, performance_metrics


def plot_embedding_process(classifier_name: str) -> go.Figure:
    """Visualize the embedding process for a classifier"""
    fig = go.Figure()
    
    info = CLASSIFIERS.get(classifier_name, {})
    data_type = info.get("data_type", "Unknown")
    color = info.get("color", "#1f77b4")
    
    # Simulate embedding process
    if "MsaComposition" in data_type:
        steps = ["Raw Sequences", "Tokenization", "MSA Composition", "Features"]
        values = [100, 95, 20, 20]
    elif "SiteComposition" in data_type:
        steps = ["Raw Sequences", "Tokenization", "Site Composition", "Features"]
        values = [100, 95, 80, 21]
    elif "Sequences" in data_type:
        steps = ["Raw Sequences", "Tokenization", "Embedding", "Attention"]
        values = [100, 95, 90, 50]
    else:
        steps = ["Raw Sequences", "Tokenization", "Embedding", "Features"]
        values = [100, 80, 60, 40]
    
    fig.add_trace(go.Bar(
        x=steps,
        y=values,
        marker_color=color,
        text=[f"{v}%" for v in values],
        textposition="outside",
        name="Transformation"
    ))
    
    fig.update_layout(
        title=f"Embedding Process - {classifier_name}<br><sub>{data_type}</sub>",
        xaxis_title="Transformation Steps",
        yaxis_title="Relative Dimensionality (%)",
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def visualize_tokenization_example(sequence: str = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDTSIHFDLAKAYLDTLIEIAPTRIRAGFGGNLQSRLLKKADKVVIMVSHRSGETEDTFIADLVVGLCTGQIKTGAPCRSERLAKYNQLMRIEKDYVKQ") -> str:
    """Visualize a tokenization example"""
    tokens = []
    token_map = {}
    token_id = 1
    
    for aa in sequence[:50]:
        if aa not in token_map:
            token_map[aa] = token_id
            token_id += 1
        tokens.append(token_map[aa])
    
    html = f"""
    <div style="font-family: monospace; background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem;">
        <h4>Tokenization Example</h4>
        <p><strong>Original Sequence:</strong></p>
        <p style="font-size: 0.9rem; word-break: break-all;">{sequence[:50]}...</p>
        <p><strong>Tokens:</strong></p>
        <p style="font-size: 0.9rem;">{tokens}</p>
        <p><strong>Mapping:</strong></p>
        <p style="font-size: 0.8rem;">{dict(list(token_map.items())[:10])}</p>
    </div>
    """
    return html


def plot_training_curves(history_df: pd.DataFrame, classifier_name: str) -> go.Figure:
    """Plot training curves"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Training Loss", "Validation Loss", 
                       "Validation Accuracy", "F1 Score"),
        vertical_spacing=0.12
    )
    
    color = CLASSIFIERS.get(classifier_name, {}).get("color", "#1f77b4")
    
    if "epoch" in history_df.columns:
        epochs = history_df["epoch"]
        
        if "train_loss" in history_df.columns:
            fig.add_trace(
                go.Scatter(x=epochs, y=history_df["train_loss"], 
                          mode="lines", name="Train Loss", line=dict(color=color)),
                row=1, col=1
            )
        
        if "val_loss" in history_df.columns:
            fig.add_trace(
                go.Scatter(x=epochs, y=history_df["val_loss"], 
                          mode="lines", name="Val Loss", line=dict(color=color, dash="dash")),
                row=1, col=2
            )
        
        if "val_acc" in history_df.columns:
            fig.add_trace(
                go.Scatter(x=epochs, y=history_df["val_acc"], 
                          mode="lines", name="Val Acc", line=dict(color=color)),
                row=2, col=1
            )
        
        if "f1" in history_df.columns or "f1_score" in history_df.columns:
            f1_col = "f1" if "f1" in history_df.columns else "f1_score"
            fig.add_trace(
                go.Scatter(x=epochs, y=history_df[f1_col], 
                          mode="lines", name="F1 Score", line=dict(color=color)),
                row=2, col=2
            )
    
    fig.update_layout(
        title=f"Training Curves - {classifier_name}",
        height=600,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig


def plot_roc_curve(roc_df: pd.DataFrame, classifier_name: str) -> go.Figure:
    """Plot ROC curve"""
    fig = go.Figure()
    
    color = CLASSIFIERS.get(classifier_name, {}).get("color", "#1f77b4")
    
    if "fpr" in roc_df.columns and "tpr" in roc_df.columns:
        fig.add_trace(go.Scatter(
            x=roc_df["fpr"],
            y=roc_df["tpr"],
            mode="lines",
            name=f"ROC - {classifier_name}",
            line=dict(color=color, width=3),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"
        ))
        
        # Diagonal line (random)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash"),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"ROC Curve - {classifier_name}",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        height=500,
        template="plotly_white",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def plot_prediction_distribution(pred_df: pd.DataFrame, classifier_name: str) -> go.Figure:
    """Visualize prediction distribution"""
    fig = go.Figure()
    
    color = CLASSIFIERS.get(classifier_name, {}).get("color", "#1f77b4")
    
    if "prob_real" in pred_df.columns:
        fig.add_trace(go.Histogram(
            x=pred_df["prob_real"],
            nbinsx=50,
            name="Probability Distribution",
            marker_color=color,
            opacity=0.7
        ))
    
    fig.update_layout(
        title=f"Prediction Distribution - {classifier_name}",
        xaxis_title="Probability of being Real",
        yaxis_title="Frequency",
        height=400,
        template="plotly_white"
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Phylogenetic Classifiers Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        real_path = st.text_input(
            "Real Data Path",
            value=str(DEFAULT_REAL_PATH),
            help="Directory containing real alignments"
        )
        
        sim_path = st.text_input(
            "Simulated Data Path",
            value=str(DEFAULT_SIM_PATH),
            help="Directory containing simulated alignments"
        )
        
        results_path = st.text_input(
            "Classification Results Path",
            value=str(DEFAULT_RESULTS_PATH),
            help="Directory containing classification results"
        )
        
        run_number = st.selectbox(
            "Run",
            options=[1, 2],
            index=0,
            help="Run number to visualize"
        )
        
        st.markdown("---")
        st.header("📊 Display Options")
        show_real_data = st.checkbox("Show Real Data", value=True)
        show_sim_data = st.checkbox("Show Simulated Data", value=True)
        show_training = st.checkbox("Show Training", value=True)
        show_predictions = st.checkbox("Show Predictions", value=True)
        show_2d_projection = st.checkbox("Show 2D Projection", value=True)
        show_3d_projection = st.checkbox("Show 3D Projection", value=False)
    
    # Convert to Path
    real_path = Path(real_path)
    sim_path = Path(sim_path)
    results_path = Path(results_path)
    
    # ==========================================
    # SECTION 1: Data Overview
    # ==========================================
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    num_real = count_fasta_files(real_path)
    num_sim = count_fasta_files(sim_path)
    
    with col1:
        st.metric("Real Alignments", num_real)
    
    with col2:
        st.metric("Simulated Alignments", num_sim)
    
    with col3:
        total = num_real + num_sim
        st.metric("Total", total)
    
    with col4:
        ratio = (num_sim / num_real * 100) if num_real > 0 else 0
        st.metric("Sim/Real Ratio", f"{ratio:.1f}%")
    
    # Distribution chart
    if show_real_data or show_sim_data:
        fig_data = go.Figure(data=[
            go.Bar(x=["Real", "Simulated"], 
                  y=[num_real, num_sim],
                  marker_color=["#2E86AB", "#A23B72"],
                  text=[num_real, num_sim],
                  textposition="outside")
        ])
        fig_data.update_layout(
            title="Data Distribution",
            height=300,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig_data, use_container_width=True)
    
    # Example sequences
    if show_real_data and real_path.exists():
        st.subheader("🔬 Example Real Sequences")
        real_files = list(real_path.glob("*.fasta"))[:3]
        if real_files:
            for fasta_file in real_files:
                with st.expander(f"📄 {fasta_file.name}"):
                    sequences = load_fasta_sample(fasta_file, max_seqs=3)
                    for seq in sequences:
                        st.code(f">{seq['id']}\n{seq['seq']}", language=None)
    
    # Real vs Simulated comparison
    if show_real_data and show_sim_data and real_path.exists() and sim_path.exists():
        st.subheader("🔄 Real vs Simulated Alignment Examples")
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <p><strong>These examples show simulated alignments generated from real alignments.</strong></p>
            <p>Compare the sequences to see how simulations preserve structure while introducing variation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        pairs = find_matching_real_sim_pairs(real_path, sim_path, max_pairs=5)
        
        if pairs:
            for real_file, sim_file in pairs:
                with st.expander(f"📊 {real_file.stem} - Real vs Simulated", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔵 Real Alignment**")
                        real_seqs = load_fasta_sample(real_file, max_seqs=5)
                        if real_seqs:
                            for seq in real_seqs:
                                st.code(f">{seq['id']}\n{seq['seq']}", language=None)
                        else:
                            st.info("No sequences found")
                    
                    with col2:
                        st.markdown("**🟣 Simulated Alignment**")
                        sim_seqs = load_fasta_sample(sim_file, max_seqs=5)
                        if sim_seqs:
                            for seq in sim_seqs:
                                st.code(f">{seq['id']}\n{seq['seq']}", language=None)
                        else:
                            st.info("No sequences found")
    
    # ==========================================
    # SECTION 2: Classifiers
    # ==========================================
    st.header("🤖 Classifiers")
    
    selected_classifier = st.selectbox(
        "Select a Classifier",
        options=list(CLASSIFIERS.keys()),
        index=0
    )
    
    classifier_info = CLASSIFIERS[selected_classifier]
    
    # Classifier information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="classifier-card">
            <h3>{selected_classifier}</h3>
            <p><strong>Type:</strong> {classifier_info['type']}</p>
            <p><strong>Data Type:</strong> {classifier_info['data_type']}</p>
            <p><strong>Description:</strong> {classifier_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: {classifier_info['color']}; 
                    height: 100px; border-radius: 0.5rem; 
                    display: flex; align-items: center; justify-content: center;">
            <span style="color: white; font-size: 1.5rem; font-weight: bold;">
                {classifier_info['type']}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Embedding process
    st.subheader("🔄 Embedding Process")
    fig_embedding = plot_embedding_process(selected_classifier)
    st.plotly_chart(fig_embedding, use_container_width=True)
    
    # Tokenization example
    with st.expander("🔤 Tokenization Example"):
        example_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDTSIHFDLAKAYLDTLIEIAPTRIRAGFGGNLQSRLLKKADKVVIMVSHRSGETEDTFIADLVVGLCTGQIKTGAPCRSERLAKYNQLMRIEKDYVKQ"
        tokenization_html = visualize_tokenization_example(example_seq)
        st.markdown(tokenization_html, unsafe_allow_html=True)
    
    # ==========================================
    # SECTION 3: 2D/3D Projections
    # ==========================================
    if (show_2d_projection or show_3d_projection) and real_path.exists() and sim_path.exists():
        st.header("🗺️ Feature Space Projections & Decision Boundaries")
        
        # Projection settings
        col_proj1, col_proj2, col_proj3 = st.columns([1, 1, 2])
        with col_proj1:
            projection_method = st.selectbox(
                "Projection Method",
                options=["t-SNE", "PCA"],
                index=0,
                help="t-SNE: Non-linear (recommended). PCA: Linear."
            )
        
        with col_proj2:
            n_components = 3 if show_3d_projection else 2
            st.info(f"Dimensions: {n_components}D")
        
        # Story narrative
        st.markdown("""
        <div class="story-box">
            <h3 style="color: white; margin-top: 0;">📖 The Classification Story</h3>
            <p style="color: white; font-size: 1.1rem;">
            Our classifiers learn to distinguish between real and simulated alignments by finding 
            patterns in high-dimensional feature space. By projecting these features into 2D or 3D, 
            we can visualize how well each classifier separates the two classes.
            </p>
            <p style="color: white; font-size: 1.1rem;">
            <strong>🔵 Blue Zone:</strong> Real alignments (P > 0.5)<br>
            <strong>🟣 Purple Zone:</strong> Simulated alignments (P < 0.5)<br>
            <strong>⚫ Black Line:</strong> Decision boundary (P = 0.5)<br>
            <strong>❌ Red/Orange X:</strong> Misclassifications
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Extract features and compute projections
        with st.spinner(f"Computing {n_components}D {projection_method} projection..."):
            try:
                features, labels, filenames = extract_features_from_data(
                    real_path, sim_path, selected_classifier, 
                    max_samples=500 if projection_method == "t-SNE" else 1000
                )
                
                original_features = features.copy() if selected_classifier == "LogisticRegressionClassifier" else None
                
                # Compute projection
                if projection_method == "t-SNE":
                    features_proj, tsne, sample_indices = compute_tsne_projection(
                        features, n_components=n_components, perplexity=30.0
                    )
                    pca = None
                    scaler = None
                    if sample_indices is not None:
                        labels = labels[sample_indices]
                        filenames = [filenames[i] for i in sample_indices]
                        if original_features is not None:
                            original_features = original_features[sample_indices]
                else:  # PCA
                    features_proj, pca, scaler = compute_pca_projection(features, n_components=n_components)
                    tsne = None
                    sample_indices = None
                
                # Load predictions
                model = load_classifier_model(results_path, selected_classifier, run_number)
                predictions = None
                if results_path.exists():
                    pred_df = load_predictions(results_path, selected_classifier, run_number)
                    if pred_df is not None and "prob_real" in pred_df.columns:
                        pred_dict = dict(zip(pred_df["filename"], pred_df["prob_real"]))
                        predictions = np.array([pred_dict.get(f, 0.5) for f in filenames])
                
                # Plot 2D
                if show_2d_projection and n_components >= 2:
                    st.subheader("📊 2D Projection")
                    fig_2d, perf_metrics_2d = plot_2d_separation(
                        features_proj[:, :2], labels, selected_classifier,
                        original_features=original_features,
                        pca=pca,
                        scaler=scaler,
                        predictions=predictions,
                        model=model,
                        projection_type=projection_method,
                        filenames=filenames
                    )
                    
                    if projection_method == "PCA" and pca is not None:
                        variance = pca.explained_variance_ratio_
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Variance", f"{variance[:2].sum()*100:.1f}%")
                        with col2:
                            st.metric("PC1 Variance", f"{variance[0]*100:.1f}%")
                        with col3:
                            st.metric("PC2 Variance", f"{variance[1]*100:.1f}%")
                    
                    st.plotly_chart(fig_2d, use_container_width=True)
                    
                    # Performance metrics
                    if perf_metrics_2d.get('accuracy') is not None:
                        st.subheader("📈 Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{perf_metrics_2d['accuracy']:.1%}")
                        with col2:
                            st.metric("Error Rate", f"{perf_metrics_2d['error_rate']:.1%}")
                        with col3:
                            st.metric("Total Errors", f"{perf_metrics_2d['n_errors']}/{len(labels)}")
                        with col4:
                            correct = len(labels) - perf_metrics_2d['n_errors']
                            st.metric("Correct", f"{correct}/{len(labels)}")
                
                # Plot 3D
                if show_3d_projection and n_components == 3:
                    st.subheader("🎯 3D Projection")
                    fig_3d, perf_metrics_3d = plot_3d_separation(
                        features_proj, labels, selected_classifier,
                        original_features=original_features,
                        pca=pca,
                        scaler=scaler,
                        predictions=predictions,
                        projection_type=projection_method,
                        filenames=filenames
                    )
                    
                    if projection_method == "PCA" and pca is not None:
                        variance = pca.explained_variance_ratio_
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Variance", f"{variance.sum()*100:.1f}%")
                        with col2:
                            st.metric("PC1", f"{variance[0]*100:.1f}%")
                        with col3:
                            st.metric("PC2", f"{variance[1]*100:.1f}%")
                        with col4:
                            st.metric("PC3", f"{variance[2]*100:.1f}%")
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Performance metrics
                    if perf_metrics_3d.get('accuracy') is not None:
                        st.subheader("📈 Performance Metrics (3D)")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{perf_metrics_3d['accuracy']:.1%}")
                        with col2:
                            st.metric("Error Rate", f"{perf_metrics_3d['error_rate']:.1%}")
                        with col3:
                            st.metric("Total Errors", f"{perf_metrics_3d['n_errors']}/{len(labels)}")
                        with col4:
                            correct = len(labels) - perf_metrics_3d['n_errors']
                            st.metric("Correct", f"{correct}/{len(labels)}")
                
                # Data statistics
                st.subheader("📊 Data Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", len(labels))
                with col2:
                    st.metric("Real", int((labels == 1).sum()))
                with col3:
                    st.metric("Simulated", int((labels == 0).sum()))
                with col4:
                    st.metric("Feature Dims", features.shape[1] if len(features.shape) > 1 else "N/A")
            
            except Exception as e:
                st.error(f"Error computing projection: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # ==========================================
    # SECTION 4: Training Results
    # ==========================================
    if show_training and results_path.exists():
        st.header("📊 Training Results")
        
        history_df = load_training_history(results_path, selected_classifier, run_number)
        
        if history_df is not None and not history_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if "val_loss" in history_df.columns:
                    st.metric("Best Val Loss", f"{history_df['val_loss'].min():.4f}")
            
            with col2:
                if "val_acc" in history_df.columns:
                    st.metric("Best Val Acc", f"{history_df['val_acc'].max():.4f}")
            
            with col3:
                f1_col = "f1" if "f1" in history_df.columns else "f1_score"
                if f1_col in history_df.columns:
                    st.metric("Best F1", f"{history_df[f1_col].max():.4f}")
            
            with col4:
                if "epoch" in history_df.columns:
                    st.metric("Epochs", int(history_df["epoch"].max()))
            
            fig_training = plot_training_curves(history_df, selected_classifier)
            st.plotly_chart(fig_training, use_container_width=True)
        else:
            st.info(f"⚠️ No training data for {selected_classifier} (Run {run_number})")
    
    # ==========================================
    # SECTION 5: Predictions
    # ==========================================
    if show_predictions and results_path.exists():
        st.header("🎯 Predictions and Performance")
        
        pred_df = load_predictions(results_path, selected_classifier, run_number)
        roc_df = load_roc_data(results_path, selected_classifier, run_number)
        
        if pred_df is not None and not pred_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "prob_real" in pred_df.columns:
                    st.metric("Mean Probability", f"{pred_df['prob_real'].mean():.4f}")
            
            with col2:
                if "pred_class" in pred_df.columns:
                    st.metric("Predicted Real", (pred_df["pred_class"] == 1).sum())
            
            with col3:
                st.metric("Total Predictions", len(pred_df))
            
            st.subheader("📊 Prediction Distribution")
            fig_pred = plot_prediction_distribution(pred_df, selected_classifier)
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info(f"⚠️ No predictions for {selected_classifier} (Run {run_number})")
        
        if roc_df is not None and not roc_df.empty:
            st.subheader("📈 ROC Curve")
            fig_roc = plot_roc_curve(roc_df, selected_classifier)
            st.plotly_chart(fig_roc, use_container_width=True)
    
    # ==========================================
    # SECTION 6: Classifier Comparison
    # ==========================================
    if results_path.exists():
        st.header("⚖️ Classifier Comparison")
        
        comparison_data = []
        
        for clf_name in CLASSIFIERS.keys():
            history_df = load_training_history(results_path, clf_name, run_number)
            roc_df = load_roc_data(results_path, clf_name, run_number)
            
            metrics = {"classifier": clf_name}
            
            if history_df is not None and not history_df.empty:
                if "val_acc" in history_df.columns:
                    metrics["best_val_acc"] = history_df["val_acc"].max()
                f1_col = "f1" if "f1" in history_df.columns else "f1_score"
                if f1_col in history_df.columns:
                    metrics["best_f1"] = history_df[f1_col].max()
            
            if roc_df is not None and not roc_df.empty:
                if "tpr" in roc_df.columns and "fpr" in roc_df.columns:
                    auc = np.trapz(roc_df["tpr"], roc_df["fpr"])
                    metrics["auc"] = auc
            
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            fig_comparison = go.Figure()
            
            if "best_val_acc" in comparison_df.columns:
                fig_comparison.add_trace(go.Bar(
                    x=comparison_df["classifier"],
                    y=comparison_df["best_val_acc"],
                    name="Val Accuracy",
                    marker_color="#2E86AB"
                ))
            
            if "best_f1" in comparison_df.columns:
                fig_comparison.add_trace(go.Bar(
                    x=comparison_df["classifier"],
                    y=comparison_df["best_f1"],
                    name="F1 Score",
                    marker_color="#A23B72"
                ))
            
            fig_comparison.update_layout(
                title="Performance Comparison",
                xaxis_title="Classifier",
                yaxis_title="Score",
                height=400,
                template="plotly_white",
                barmode="group"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.subheader("📋 Comparison Table")
            st.dataframe(comparison_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🧬 Phylogenetic Classification Dashboard - Enhanced Edition</p>
        <p>With 2D/3D Decision Boundary Visualization</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()