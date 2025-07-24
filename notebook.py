import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full", app_title="Code For FantasyID Evaluation")


@app.cell
def _(MODELS, get_df, metrics_df, mo, px, roc_df, split, table_format):
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


    def hist_plot(split):
        df = get_df(model=None, split=split)
        df.label = df.label.replace(
            {
                "face": "attack",
                "text": "attack",
            }
        )
        df.model = df.model.replace(MODELS)
        fig = px.strip(
            df,
            x="model",
            y="score",
            color="label",
            labels=dict(model="Model"),
            # box=True,
            # points="all",
        )
        return fig

    _save_format = "svg"
    _roc_plot = mo.vstack(
        [
            split,
            mo.hstack(
                [
                    mo.ui.plotly(
                        roc_plot(split.value),
                        config={
                            "toImageButtonOptions": {
                                "format": _save_format,
                                "filename": f"roc_{split.value}",
                                "height": 240 * 1.2,
                                "width": 480 * 1.2,
                            }
                        },
                    ),
                    mo.ui.plotly(
                        hist_plot(split.value),
                        config={
                            "toImageButtonOptions": {
                                "format": _save_format,
                                "filename": f"hist_{split.value}",
                                "height": 420,
                                "width": 480,
                            },
                        },
                    ),
                ]
            ),
        ]
    )
    _table = mo.vstack(
        [
            table_format,
            mo.md(
                metrics_df.to_markdown(index=False)
                if table_format.value == "markdown"
                else "```\n" + metrics_df.to_latex(index=False, float_format="%.1f") + "\n```"
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
    import tabulate
    import pyarrow

    pio.templates.default = "plotly_white"
    MODELS = {
        "trufor": "TruFor",
        "mmfusion": "MMFusion",
        "unifd": "UniFD",
        "fatformer": "FatFormer",
    }
    return MODELS, mo, pd, px, sklearn


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
    table_format = mo.ui.dropdown(
        options=("markdown", "latex"), value="markdown", label="Table Format"
    )
    return split, table_format


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


if __name__ == "__main__":
    app.run()
