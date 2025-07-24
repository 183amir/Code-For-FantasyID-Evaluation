import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full", app_title="Code For FantasyID Evaluation")


@app.cell
def _(
    MODELS,
    format,
    hist_plot,
    metrics_df,
    mo,
    px,
    roc_df,
    split,
    table_format,
):
    def roc_plot(split):
        df = roc_df(split)
        df.model = df.model.replace(MODELS)
        fig = px.line(
            df,
            x="fpr",
            y="tpr",
            color="model",
            labels=dict(
                fpr="False Positive Rate", tpr="True Positive Rate", model="Model"
            ),
            color_discrete_sequence=px.colors.qualitative.D3,
        )
        fig.update_layout(
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
        )
        return fig

    _roc_plot = mo.vstack(
        [
            mo.hstack([split, format]),
            mo.vstack(
                [
                    mo.ui.plotly(
                        roc_plot(split.value),
                        config={
                            "toImageButtonOptions": {
                                "format": format.value,
                                "filename": f"roc_{split.value}",
                                "height": 240 * 1.2,
                                "width": 480 * 1.2,
                            }
                        },
                    ),
                    mo.hstack([split, format]),
                    mo.ui.plotly(
                        hist_plot(split.value),
                        config={
                            "toImageButtonOptions": {
                                "format": format.value,
                                "filename": f"hist_{split.value}",
                                "height": 420,
                                "width": 480,
                                "scale": 2,
                            },
                        },
                    ),
                ]
            ),
        ]
    )
    _table = mo.vstack(
        [
            # table_format,
            mo.md(
                metrics_df.to_markdown(index=False)
                if table_format.value == "markdown"
                else "```\n"
                + metrics_df.to_latex(index=False, float_format="%.1f")
                + "\n```"
            ),
        ]
    )
    mo.md(f"""

    {mo.hstack([_roc_plot, _table], widths=[3, 1])}
    """)
    return


@app.cell
def _(metrics_df, mo):
    mo.md("```\n" + metrics_df.to_latex(index=False, float_format="%.1f") + "\n```")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import sklearn
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates.default = "plotly_white"
    MODELS = {
        "trufor": "TruFor",
        "mmfusion": "MMFusion",
        "unifd": "UniFD",
        "fatformer": "FatFormer",
    }
    return MODELS, go, mo, pd, px, sklearn


@app.cell
def _(MODELS, pd):
    def df_of_model(model):
        df = pd.read_parquet(
            f"https://raw.githubusercontent.com/183amir/Code-For-FantasyID-Evaluation/refs/heads/main/scores/{model}/detection.parquet",
            columns=("label", "score"),
            filters=[("protocol", "==", "test")],
        )
        df["model"] = model
        df["label"] = df["label"].replace(
            {
                "facedancer": "face",
                "digital_3": "text",
                "textdiffuserft_bfei": "text",
            }
        )
        return df

    df_all_models = pd.concat([df_of_model(m) for m in MODELS])
    print(df_all_models)
    return (df_all_models,)


@app.cell
def _(df_all_models):
    print(df_all_models.label.value_counts())
    return


@app.cell
def _(MODELS, df_all_models, pd, sklearn):
    def get_df(model=None, split="all"):
        if model is not None:
            df = df_all_models[df_all_models.model == model]
        else:
            df = df_all_models.copy()
        labels = ["bonafide"]
        if split == "all":
            labels.extend(["face", "text"])
        else:
            labels.append(split)
        return df[df.label.isin(labels)]

    def metrics(df):
        y_true = df["label"] == "bonafide"
        y_score = df["score"]
        y_pred = y_score >= 0.5
        return {
            "ACC": sklearn.metrics.balanced_accuracy_score(
                y_true=y_true, y_pred=y_pred
            ),
            "AUC": sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score),
            "F1": sklearn.metrics.f1_score(
                y_true=y_true, y_pred=y_pred, average="weighted"
            ),
        }

    def metrics_table():
        records = []
        for model in MODELS:
            for split in ("all", "face", "text"):
                df = get_df(model, split)
                records.append(
                    {
                        "Model": model,
                        "Protocol": split,
                        **{k: round(v * 100, 1) for k, v in metrics(df).items()},
                    }
                )
        df = pd.DataFrame(records)
        df.Model = df.Model.replace(MODELS)
        return df

    metrics_df = metrics_table()
    print(metrics_df)
    return get_df, metrics_df


@app.cell
def _(mo):
    split = mo.ui.dropdown(
        options=("all", "face", "text"), value="all", label="Protocol"
    )
    format = mo.ui.dropdown(
        options=("png", "svg", "jpeg", "webp"),
        value="png",
        label="Download Plot Format",
    )
    table_format = mo.ui.dropdown(
        options=("markdown", "latex"), value="markdown", label="Table Format"
    )
    return format, split, table_format


@app.cell
def _(MODELS, get_df, pd, sklearn, split):
    def roc_df(split):
        records = []
        for model in MODELS:
            df = get_df(model, split)
            y_true = df["label"] == "bonafide"
            y_score = df["score"]
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
            df_roc = pd.DataFrame(
                {
                    "fpr": fpr,
                    "tpr": tpr,
                }
            )
            df_roc["model"] = model
            records.append(df_roc)
        return pd.concat(records)

    print(roc_df(split.value))
    return (roc_df,)


@app.cell
def _(MODELS, get_df, go):
    def hist_plot(split):
        df = get_df(model=None, split=split)
        df.label = df.label.replace(
            {
                "face": "attack",
                "text": "attack",
            }
        )
        df.model = df.model.replace(MODELS)
        unique_models = df["model"].unique()

        # Define colors based on matplotlib's 'tab' palette with opacity
        color_map = {
            "bonafide": {
                "fill": "rgba(31, 119, 180, 0.6)",
                "line": "rgb(31, 119, 180)",
            },
            "attack": {"fill": "rgba(255, 127, 14, 0.6)", "line": "rgb(255, 127, 14)"},
        }

        legend_map = {"bonafide": "Bonafide", "attack": "Attacks"}

        # 2. Initialize a Figure object
        fig = go.Figure()

        # 3. Loop through each model to create split violin traces
        for i, model_name in enumerate(unique_models):
            # --- Bonafide Trace (Right Side) ---
            df_bonafide = df[(df["model"] == model_name) & (df["label"] == "bonafide")]
            fig.add_trace(
                go.Violin(
                    x=df_bonafide["model"],
                    y=df_bonafide["score"],
                    scalegroup=model_name,
                    name=legend_map["bonafide"],
                    legendgroup=legend_map["bonafide"],
                    side="positive",  # Swapped to 'positive' (right) to match Matplotlib
                    points=False,
                    box_visible=False,
                    meanline_visible=False,
                    line_color=color_map["bonafide"]["line"],
                    fillcolor=color_map["bonafide"]["fill"],
                    showlegend=(i == 0),
                )
            )
            # --- Attack Trace (Left Side) ---
            df_attack = df[(df["model"] == model_name) & (df["label"] == "attack")]
            fig.add_trace(
                go.Violin(
                    x=df_attack["model"],
                    y=df_attack["score"],
                    scalegroup=model_name,
                    name=legend_map["attack"],
                    legendgroup=legend_map["attack"],
                    side="negative",  # Swapped to 'negative' (left) to match Matplotlib
                    points=False,
                    box_visible=False,
                    meanline_visible=False,
                    line_color=color_map["attack"]["line"],
                    fillcolor=color_map["attack"]["fill"],
                    showlegend=(i == 0),
                )
            )

        # 4. Update traces and layout to match the target image
        fig.update_traces(width=0.8, box_line_width=1)

        fig.update_layout(
            violinmode="overlay",
            violingap=0,
            # plot_bgcolor='white',
            yaxis=dict(
                range=[0, 1],
                #     gridcolor='lightgrey',
                #     gridwidth=1,
                #     griddash='dash',
                #     showline=True,
                #     linewidth=1,
                #     linecolor='black',
                #     mirror=True,
                #     zeroline=False
            ),
            # xaxis=dict(
            #     showgrid=False,
            #     showline=True,
            #     linewidth=1,
            #     linecolor='black',
            #     mirror=True
            # ),
            title_text=None,
            xaxis_title=None,
            yaxis_title=None,
            legend=dict(
                x=0.98,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                # bordercolor='black',
                # borderwidth=1
            ),
        )

        return fig

    return (hist_plot,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
