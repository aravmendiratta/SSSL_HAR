"""
Visualization utilities for representation learning analysis and comparison charting.
Generates t-SNE scatter plots matching Figure 3 and robustness column bar charts matching Figures 4-6.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Union


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    title: str = "t-SNE Visualization of Activity Feature Embeddings",
    use_plotly: bool = True,
    save_path: Optional[str] = None
):
    """
    Performs t-SNE dimensionality reduction on learned 256-D aggregated embeddings and plots clusters by activity class.
    
    Args:
        embeddings: Feature matrix of shape (N_samples, D=256 or 64).
        labels: Integer activity class indices of shape (N_samples,).
        class_names: Mapping of indices to activity text labels.
        title: Plot header title.
        use_plotly: If True, returns an interactive Plotly Figure for Streamlit dashboard display.
        save_path: Optional filesystem filepath to export image artifact.
    """
    n_samples = embeddings.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    tsne_coords = tsne.fit_transform(embeddings)
    
    labels_arr = np.asarray(labels)
    if class_names is not None:
        legend_labels = [class_names[idx] if idx < len(class_names) else f"Class {idx}" for idx in labels_arr]
    else:
        legend_labels = [f"Activity {idx}" for idx in labels_arr]
        
    if use_plotly:
        fig = px.scatter(
            x=tsne_coords[:, 0],
            y=tsne_coords[:, 1],
            color=legend_labels,
            title=title,
            labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2", "color": "Activity Classes"},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(20, 24, 33, 1)",
            plot_bgcolor="rgba(15, 18, 25, 1)",
            font=dict(family="Inter, Roboto, sans-serif", size=13, color="#E0E6ED"),
            hoverlabel=dict(bgcolor="rgba(30, 35, 45, 0.95)", font_size=13),
            margin=dict(l=40, r=40, t=60, b=40),
            legend_title_text="Activity Classes"
        )
        fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.2)")))
        if save_path:
            fig.write_image(save_path)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = sorted(list(set(legend_labels)))
        cmap = plt.get_cmap("tab20")
        for idx, lbl in enumerate(unique_labels):
            mask = np.array(legend_labels) == lbl
            ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], label=lbl, alpha=0.8, s=40, edgecolors="none")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Activity Classes")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
        return fig


def generate_comparison_chart(
    data_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ["Acc", "F1_M", "F1_W"],
    title: str = "Performance Comparison across Configurations",
    x_title: str = "Experimental Configuration",
    y_title: str = "Scores (%)"
):
    """
    Generates interactive grouped bar charts comparing performance metrics across conditions
    (e.g., Figures 4, 5, 6 for sampling rate matching and sensor placement mismatches).
    """
    configs = list(data_dict.keys())
    fig = go.Figure()
    
    colors = {"Acc": "#3B82F6", "F1_M": "#10B981", "F1_W": "#EF4444", "Recall": "#F59E0B", "Prec": "#8B5CF6"}
    
    for metric in metrics_to_plot:
        y_vals = [data_dict[cfg].get(metric, 0.0) for cfg in configs]
        fig.add_trace(go.Bar(
            name=metric,
            x=configs,
            y=y_vals,
            marker_color=colors.get(metric, "#6366F1"),
            text=[f"{v:.1f}%" for v in y_vals],
            textposition="auto",
        ))
        
    fig.update_layout(
        barmode="group",
        title=dict(text=title, font=dict(size=18, family="Outfit, Inter, sans-serif")),
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis=dict(range=[50, 100], gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark",
        paper_bgcolor="rgba(20, 24, 33, 1)",
        plot_bgcolor="rgba(15, 18, 25, 1)",
        legend=dict(title="Metrics", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, Roboto, sans-serif", size=13, color="#E0E6ED"),
        margin=dict(l=40, r=40, t=70, b=40)
    )
    return fig
