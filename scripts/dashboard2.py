# scripts/dashboard.py
# Dash dashboard complet avec callbacks et auto-refresh.
import io
import base64
from pathlib import Path
import pandas as pd
from dash import Dash, dcc, html, Output, Input, State, callback_context, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import flask
import traceback

# Optional tree rendering (if available)
try:
    from Bio import Phylo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PHYLO = True
except Exception:
    HAS_PHYLO = False

REFRESH_INTERVAL_MS = 5000  # auto-refresh every 5s


def get_project_root():
    """Return project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def results_path():
    return get_project_root() / "results"


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def list_tree_files():
    td = results_path() / "trees"
    if not td.exists():
        return []
    return sorted([p for p in td.glob("*") if p.is_file()])


def make_app():
    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    server = app.server

    app.layout = dbc.Container(
        [
            html.H2("Pipeline Dashboard", className="my-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Source / Controls"),
                                    dbc.CardBody(
                                        [
                                            html.Div("Base results dir:"),
                                            html.Pre(str(results_path())),
                                            dbc.Button("Refresh now", id="btn-refresh", color="primary", className="mb-2"),
                                            html.Div(id="last-action", style={"fontSize": "12px", "color": "gray"}),
                                            html.Hr(),
                                            html.Div("Auto-refresh interval (ms):"),
                                            dcc.Input(id="input-interval", type="number", value=REFRESH_INTERVAL_MS, step=1000),
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Files"),
                                    dbc.CardBody(
                                        [
                                            html.Div("Available metric file:"),
                                            html.Ul(id="metrics-file"),
                                            html.Hr(),
                                            html.Div("Available classification file:"),
                                            html.Ul(id="classif-file"),
                                            html.Hr(),
                                            html.Div("Tree files (click to view):"),
                                            html.Ul(id="tree-list"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs-main",
                                value="tab-sim",
                                children=[
                                    dcc.Tab(label="Simulation (MPD)", value="tab-sim"),
                                    dcc.Tab(label="Classification", value="tab-classif"),
                                    dcc.Tab(label="Selected Tree", value="tab-tree"),
                                    dcc.Tab(label="Logs", value="tab-logs"),
                                ],
                            ),
                            html.Div(id="tab-content", className="mt-3"),
                        ],
                        width=9,
                    ),
                ]
            ),
            # Interval for auto-refresh
            dcc.Interval(id="interval-refresh", interval=REFRESH_INTERVAL_MS, n_intervals=0),
            # Hidden store to carry selected tree path
            dcc.Store(id="selected-tree", data=""),
        ],
        fluid=True,
    )

    # -------------------------
    # Callbacks
    # -------------------------

    @app.callback(
        Output("interval-refresh", "interval"),
        Input("input-interval", "value"),
    )
    def update_interval(ms):
        try:
            ms = int(ms)
            return max(1000, ms)
        except Exception:
            return REFRESH_INTERVAL_MS

    @app.callback(
        Output("metrics-file", "children"),
        Output("classif-file", "children"),
        Output("tree-list", "children"),
        Input("interval-refresh", "n_intervals"),
        Input("btn-refresh", "n_clicks"),
    )
    def refresh_file_lists(n_intervals, n_clicks):
        """List available files in the left column; updates periodically or on click."""
        try:
            # Metrics
            metrics_file = results_path() / "metrics" / "phylo_metrics.csv"
            mf_node = html.Li(str(metrics_file)) if metrics_file.exists() else html.Li("No metrics file found")

            # Classification (attempt common filenames)
            cd = results_path() / "classification"
            classif_candidates = []
            if cd.exists():
                for candidate in sorted(cd.glob("*.csv")):
                    classif_candidates.append(html.Li(str(candidate)))
            if not classif_candidates:
                classif_candidates = [html.Li("No classification CSVs found")]

            # Trees
            trees = list_tree_files()
            if trees:
                tree_nodes = [
                    html.Li(
                        html.Button(p.name, n_clicks=0, id={"type": "tree-btn", "index": str(p.name)}, style={"border": "none", "background": "none", "color": "blue", "textDecoration": "underline", "cursor": "pointer"})
                    ) for p in trees
                ]
            else:
                tree_nodes = [html.Li("No tree files found in results/trees")]

            return mf_node, classif_candidates, tree_nodes

        except Exception as e:
            tb = traceback.format_exc()
            return html.Li("Error listing files: " + str(e)), html.Li(""), html.Li(tb)

    @app.callback(
        Output("selected-tree", "data"),
        Output("last-action", "children"),
        Input({"type": "tree-btn", "index": ALL}, "n_clicks"),
        State({"type": "tree-btn", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def pick_tree(n_clicks_list, ids):
        """When a tree name is clicked, store it in dcc.Store."""
        ctx = callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trig = ctx.triggered[0]["prop_id"]
        # find which button triggered
        for idx, n in enumerate(n_clicks_list):
            if n and ids and idx < len(ids):
                name = ids[idx]["index"]
                # build absolute path
                p = results_path() / "trees" / name
                if p.exists():
                    return str(p), f"Selected tree: {name}"
                else:
                    return "", f"File not found: {name}"
        return "", "No selection"

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs-main", "value"),
        Input("selected-tree", "data"),
        Input("interval-refresh", "n_intervals"),
    )
    def render_tab(tab, selected_tree, _):
        """Render the main tab content depending on which tab is active."""
        try:
            if tab == "tab-sim":
                # load metrics
                metrics_file = results_path() / "metrics" / "phylo_metrics.csv"
                if metrics_file.exists():
                    df = safe_read_csv(metrics_file)
                    if df.empty:
                        return html.Div("Metrics file exists but could not be read or is empty.")
                    # build figures
                    fig_hist = px.histogram(df, x="MPD", nbins=40, title="MPD distribution")
                    fig_scatter = px.scatter(df, x="n_leaves", y="MPD", title="MPD vs n_leaves")
                    return html.Div([dcc.Graph(figure=fig_hist), dcc.Graph(figure=fig_scatter)])
                else:
                    return html.Div("No metrics file found at results/metrics/phylo_metrics.csv")

            elif tab == "tab-classif":
                classif_dir = results_path() / "classification"
                if not classif_dir.exists():
                    return html.Div("No classification directory found (results/classification).")

                # Collect all parquet + csv in subfolders and root
                files = sorted(
                    list(classif_dir.rglob("*.parquet")) +
                    list(classif_dir.rglob("*.csv")),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )

                if not files:
                    return html.Div("No classification files (.csv or .parquet) found.")

                # Most recent file
                f = files[0]

                # Load file properly
                try:
                    if f.suffix == ".csv":
                        df = safe_read_csv(f)
                    else:
                        df = pd.read_parquet(f)
                except Exception as e:
                    return html.Div(f"Could not read file {f.name}: {e}")

                if df.empty:
                    return html.Div(f"File {f.name} is empty or unreadable.")

                children = [
                    html.H4(f"Classification results — {f.name}"),
                    html.P(f"Path: {f.relative_to(results_path())}")
                ]

                # Detect probability-like column
                prob_col = None
                for c in ["probability", "prob", "score", "real_prob", "pred_prob", "y_pred"]:
                    if c in df.columns:
                        prob_col = c
                        break

                # Detect true label column
                ytrue_col = None
                for c in ["y_true", "true", "label", "target"]:
                    if c in df.columns:
                        ytrue_col = c
                        break

                # -------------------------------
                # 📊 1. Histogram of probabilities
                # -------------------------------
                if prob_col:
                    fig = px.histogram(df, x=prob_col, title=f"Distribution of {prob_col}")
                    children.append(dcc.Graph(figure=fig))

                # -------------------------------
                # 📈 2. ROC curve (if possible)
                # -------------------------------
                if prob_col and ytrue_col:
                    try:
                        from sklearn.metrics import roc_curve, auc
                        fpr, tpr, _ = roc_curve(df[ytrue_col], df[prob_col])
                        roc_auc = auc(fpr, tpr)

                        fig_roc = px.line(
                            x=fpr, y=tpr,
                            title=f"ROC curve (AUC = {roc_auc:.4f})",
                            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                        )
                        fig_roc.add_shape(
                            type="line", x0=0, x1=1, y0=0, y1=1,
                            line=dict(dash="dash")
                        )
                        children.append(dcc.Graph(figure=fig_roc))
                    except Exception as e:
                        children.append(html.Div(f"Could not compute ROC: {e}"))

                # -----------------------------------
                # 📉 3. Learning curves (train_history)
                # Detect if the loaded file is a training history
                # -----------------------------------
                is_history = "epoch" in df.columns and "loss" in df.columns

                if is_history:
                    # Loss curve
                    fig_loss = px.line(df, x="epoch", y="loss", title="Training loss")
                    children.append(dcc.Graph(figure=fig_loss))

                    # Accuracy curve
                    if "accuracy" in df.columns:
                        fig_acc = px.line(df, x="epoch", y="accuracy", title="Training accuracy")
                        children.append(dcc.Graph(figure=fig_acc))

                # -----------------------------------
                # 🧾 Table preview
                # -----------------------------------
                preview = df.head(200)
                children.append(html.H5("Preview (first 200 rows)"))
                children.append(
                    dbc.Table.from_dataframe(preview, striped=True, bordered=True, hover=True)
                )

                return html.Div(children)


            elif tab == "tab-tree":
                if not selected_tree:
                    return html.Div("No tree selected yet. Click a tree on the left to view it.")
                p = Path(selected_tree)
                if not p.exists():
                    return html.Div("Selected tree file does not exist anymore.")
                # show raw newick
                raw = p.read_text()
                content = [html.H5(f"Tree: {p.name}"), html.H6("Newick:"), html.Pre(raw[:2000])]
                # try to render an image using Bio.Phylo if available
                if HAS_PHYLO:
                    try:
                        tree = Phylo.read(str(p), "newick")
                        # draw to png in-memory
                        figfile = io.BytesIO()
                        plt.figure(figsize=(6, 6))
                        Phylo.draw(tree, do_show=False)
                        plt.savefig(figfile, format="png", bbox_inches="tight")
                        plt.close()
                        figfile.seek(0)
                        encoded = base64.b64encode(figfile.read()).decode("ascii")
                        img = html.Img(src="data:image/png;base64," + encoded, style={"maxWidth": "100%"})
                        content.append(html.Hr())
                        content.append(img)
                    except Exception as e:
                        content.append(html.Div("Could not render tree image (Bio.Phylo/Matplotlib error): " + str(e)))
                else:
                    content.append(html.Div("Tree rendering not available (install biopython + matplotlib to enable)."))
                # download link
                href = "/download-tree/" + p.name
                content.append(html.Hr())
                content.append(html.A("Download tree file", href=href, target="_blank"))
                return html.Div(content)

            elif tab == "tab-logs":
                lf = get_project_root() / "logs" / "pipeline_log.csv"
                if not lf.exists():
                    return html.Div("No logs/pipeline_log.csv found.")
                text = lf.read_text()
                return html.Div([html.H5("pipeline_log.csv"), html.Pre(text[:10000])])

            else:
                return html.Div("Tab not implemented.")
        except Exception as e:
            return html.Div("Error rendering tab: " + str(e) + "\n" + traceback.format_exc())

    # Flask route to serve tree files for download
    @server.route("/download-tree/<path:filename>")
    def download_tree(filename):
        p = results_path() / "trees" / filename
        if not p.exists():
            return flask.abort(404)
        return flask.send_file(str(p), as_attachment=True, download_name=filename)

    return app, server


def run_dashboard():
    app, server = make_app()
    app.run(debug=True)


if __name__ == "__main__":
    run_dashboard()
