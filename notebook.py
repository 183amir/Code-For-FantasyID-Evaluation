import marimo

__generated_with = "0.14.13"
app = marimo.App(width="full", app_title="Code For FantasyID Evaluation")


@app.cell
def _(
    format,
    hist_plot,
    metrics_df,
    mo,
    roc_plot,
    selected_protocol,
    selected_threshold,
    split,
    table_format,
):
    _plots = mo.vstack(
        [
            mo.hstack([split, selected_protocol, format]),
            mo.vstack(
                [
                    mo.ui.plotly(
                        roc_plot(split.value),
                        config={
                            "toImageButtonOptions": {
                                "format": format.value,
                                "filename": f"roc_{split.value.replace(' ', '_')}",
                                "height": 340,
                                "width": 490,
                                "scale": 2,
                            }
                        },
                    ),
                    mo.hstack([split, selected_protocol, format]),
                    mo.ui.plotly(
                        hist_plot(split.value),
                        config={
                            "toImageButtonOptions": {
                                "format": format.value,
                                "filename": f"hist_{split.value.replace(' ', '_')}",
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
    _table_md = mo.md(
        metrics_df.to_markdown(index=False)
        if table_format.value == "markdown"
        else "```\n" + metrics_df.to_latex(index=False, float_format="%.1f") + "\n```"
    )
    _table_latex = (
        "```\n" + metrics_df.to_latex(index=False, float_format="%.1f") + "\n```"
    )
    _images = mo.vstack(
        [
            mo.hstack(
                [
                    mo.image(
                        str(mo.notebook_location() / "public" / "chinese.png"),
                        width="50%",
                    ),
                    mo.image(
                        str(mo.notebook_location() / "public" / "turkey.png"),
                        width="50%",
                    ),
                ]
            ),
            "Examples of original digital versions of FantasyID cards.",
        ],
    )
    _header = (
        mo.md(f"""
    # <div style="text-align: center"><span style="color:blue">FantasyID</span>: A dataset for detecting digital manipulations of ID-documents</div>
    [Pavel Korshunov](<pavel.korshunov@idiap.ch>),
    [Amir Mohammadi](<amir.mohammadi@idiap.ch>),
    [Vidit Vidit](<vidit.vidit@idiap.ch>),
    [Christophe Ecabert](<christophe.ecabert@idiap.ch>),
    [Sebastien Marcel](<Sebastien.Marcel@idiap.ch>)

    Idiap Research Institute, Martigny, Switzerland

    IJCB 2025

    {_images}

    ## Summary
    <div style="text-align: justify">
    Advancements in image generation led to the availabil-
    ity of easy-to-use tools for malicious actors to create forged
    images. These tools pose a serious threat to the widespread
    Know Your Customer (KYC) applications, requiring ro-
    bust systems for detection of the forged Identity Documents
    (IDs). To facilitate the development of the detection algo-
    rithms, in this paper, we propose a novel publicly available
    (including commercial use) dataset, FantasyID, which mim-
    ics real-world IDs but without tampering with legal docu-
    ments and, compared to previous public datasets, it does
    not contain generated faces or specimen watermarks. Fan-
    tasyID contains ID cards with diverse design styles, lan-
    guages, and faces of real people. To simulate a realis-
    tic KYC scenario, the cards from FantasyID were printed
    and captured with three different devices, constituting the
    bonafide class. We have emulated digital forgery/injection
    attacks that could be performed by a malicious actor to tam-
    per the IDs using the existing generative tools. The current
    state-of-the-art forgery detection algorithms, such as Tru-
    For, MMFusion, UniFD, and FatFormer, are challenged by
    FantasyID dataset. It especially evident, in the evaluation
    conditions close to practical, with the operational thresh-
    old set on validation set so that false positive rate is at 10%,
    leading to false negative rates close to 50% across the board
    on the test set. The evaluation experiments demonstrate that
    FantasyID dataset is complex enough to be used as an eval-
    uation benchmark for detection algorithms.
    </div>
    """)
        .center()
        .style(text_align="center")
    )
    mo.md(f"""
    {mo.callout(_header, "info")}
    {_plots}
    {mo.hstack([selected_protocol, selected_threshold, "The threshold for FPR, FNR, and HTER values are fixed from the validation set."])}
    {_table_md}
    {_table_latex}
    """).center()
    return


@app.cell
def _(SPLITS, mo):
    split = mo.ui.dropdown(options=SPLITS, value="all", label="Protocol")
    return (split,)


@app.cell
def _(mo):
    format = mo.ui.dropdown(
        options=("png", "svg", "jpeg", "webp"),
        value="svg",
        label="Download Plot Format",
    )
    table_format = mo.ui.dropdown(
        options=("markdown", "latex"), value="markdown", label="Table Format"
    )
    selected_threshold = mo.ui.slider(
        start=0, stop=1, step=0.01, value=0.5, label="Threshold for ACC and F1"
    )
    selected_protocol = mo.ui.dropdown(
        options=("val", "test"), value="test", label="Group"
    )
    return format, selected_protocol, selected_threshold, table_format


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import sklearn
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import pyarrow  # noqa: F401
    import tabulate  # noqa: F401
    import jinja2  # noqa: F401

    pio.templates.default = "plotly_white"
    MODELS = {
        "trufor": "TruFor",
        "mmfusion": "MMFusion",
        "unifd": "UniFD",
        "fatformer": "FatFormer",
    }
    return MODELS, go, mo, np, pd, px, sklearn


@app.cell
def _(MODELS, pd, selected_protocol):
    def find_train_files(df):
        val_set = df["protocol"] == "val"
        last_split = df["filename"].str.split("-").str[-1]
        return last_split.str.len() <= 11 & val_set

    def df_of_model(model):
        df = pd.read_parquet(
            f"https://raw.githubusercontent.com/183amir/Code-For-FantasyID-Evaluation/refs/heads/main/public/{model}/detection.parquet",
            columns=("label", "score", "filename", "protocol"),
            # filters=[("protocol", "==", selected_protocol.value)],
        )
        train_mask = find_train_files(df)
        df.loc[train_mask, "protocol"] = "train"
        df["model"] = model
        df["label"] = df["label"].replace(
            {
                "digital_3": "attack 1",
                "facedancer": "attack 2",
                "textdiffuserft_bfei": "attack 3",
                "digital_1": "inswapper+diffste",
                "digital_2": "facedancer+textdiffuser2",
            }
        )
        return df

    df_all_models = pd.concat([df_of_model(m) for m in MODELS])
    SPLITS = ("all",) + tuple(
        set(
            df_all_models[
                df_all_models["protocol"] == selected_protocol.value
            ].label.unique()
        )
        - {"bonafide"}
    )
    print(SPLITS)
    print(df_all_models)
    return SPLITS, df_all_models


@app.cell
def _(df_all_models):
    print(df_all_models.label.value_counts())
    print(df_all_models.protocol.value_counts())
    return


@app.cell
def _(
    MODELS,
    SPLITS,
    df_all_models,
    np,
    pd,
    selected_protocol,
    selected_threshold,
    sklearn,
):
    def filter_df_by_split(df, split):
        labels = ["bonafide"]
        if split == "all":
            labels.extend(set(SPLITS) - {"all"})
        else:
            labels.append(split)
        return df[df.label.isin(labels)]

    def get_df(model=None, split="all", filter_by_protocol=True):
        if model is not None:
            df = df_all_models[df_all_models.model == model]
        else:
            df = df_all_models.copy()
        if filter_by_protocol:
            df = df[df["protocol"] == selected_protocol.value]
        if split is None:
            return df
        return filter_df_by_split(df, split)

    def far_threshold(negatives, positives, far_value=0.001, is_sorted=False):
        """from bob.measure"""
        if far_value < 0.0 or far_value > 1.0:
            raise RuntimeError("`far_value' must be in the interval [0.,1.]")

        if len(negatives) < 2:
            raise RuntimeError("the number of negative scores must be at least 2")

        epsilon = np.finfo(np.float64).eps
        scores = negatives if is_sorted else np.sort(negatives)
        if far_value >= (1 - epsilon):
            return np.nextafter(scores[0], scores[0] - 1)

        scores = np.flip(scores)
        total_count = len(scores)
        current_position = 0
        valid_threshold = np.nextafter(
            scores[current_position], scores[current_position] + 1
        )
        current_threshold = 0.0

        while current_position < total_count:
            current_threshold = scores[current_position]
            while (
                current_position < (total_count - 1)
                and scores[current_position + 1] == current_threshold
            ):
                current_position += 1
            future_far = (current_position + 1) / total_count
            if future_far > far_value:
                break
            valid_threshold = current_threshold
            current_position += 1

        return valid_threshold

    def compute_threshold(df):
        y_true = df["label"] == "bonafide"
        y_score = df["score"]
        negatives, positives = y_score[~y_true], y_score[y_true]
        return far_threshold(negatives, positives, far_value=0.1)
        # fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
        # threshold_index = np.abs(fpr - 0.01).argmin()
        # return thresholds[threshold_index]

    def metrics(model, split):
        df = get_df(model, split=None, filter_by_protocol=False)
        val_threshold = compute_threshold(df[df["protocol"] == "val"])
        df = df[df["protocol"] == selected_protocol.value]
        df = filter_df_by_split(df, split)
        y_true = df["label"] == "bonafide"
        y_score = df["score"]
        y_pred = y_score >= selected_threshold.value
        tn, fp, fn, tp = (
            sklearn.metrics.confusion_matrix(
                y_true=y_true, y_pred=y_score >= val_threshold
            )
            .ravel()
            .tolist()
        )
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        return {
            "ACC": sklearn.metrics.balanced_accuracy_score(
                y_true=y_true, y_pred=y_pred
            ),
            "AUC": sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score),
            "F1": sklearn.metrics.f1_score(
                y_true=y_true, y_pred=y_pred, average="weighted"
            ),
            "FPR": fpr,
            "FNR": fnr,
            "HTER": (fpr + fnr) / 2,
        }

    def metrics_table():
        records = []
        for model in MODELS:
            for split in SPLITS:
                records.append(
                    {
                        "Model": model,
                        "Protocol": split,
                        **{
                            k: round(v * 100, 1)
                            for k, v in metrics(model, split).items()
                        },
                    }
                )
        df = pd.DataFrame(records)
        df.Model = df.Model.replace(MODELS)
        return df

    metrics_df = metrics_table()
    print(metrics_df)
    return get_df, metrics_df


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
def _(MODELS, px, roc_df):
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
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.7)",
                title=None,
            )
        )
        return fig

    return (roc_plot,)


@app.cell
def _(MODELS, SPLITS, get_df, go):
    def hist_plot(split):
        df = get_df(model=None, split=split)
        df.label = df.label.replace({v: "attack" for v in SPLITS})
        df.model = df.model.replace(MODELS)
        unique_models = df["model"].unique()

        # Define colors based on matplotlib's 'tab' palette with opacity
        color_map = {
            "bonafide": {
                "fill": "rgba(31, 119, 180, 0.6)",
                "line": "rgb(31, 119, 180)",
            },
            "attack": {
                "fill": "rgba(255, 127, 14, 0.6)",
                "line": "rgb(255, 127, 14)",
            },
        }

        legend_map = {"bonafide": "Bonafide", "attack": "Attacks"}

        # 2. Initialize a Figure object
        fig = go.Figure()
        bandwidth = 0.04

        # 3. Loop through each model to create split violin traces
        for i, model_name in enumerate(unique_models):
            # --- Bonafide Trace (Right Side) ---
            df_bonafide = df[(df["model"] == model_name) & (df["label"] == "bonafide")]
            kw = dict(
                bandwidth=bandwidth,
                # span=[0,1],
                spanmode="hard",
                scalegroup=model_name,
                points=False,
                box_visible=False,
                meanline_visible=False,
                showlegend=(i == 0),
            )
            fig.add_trace(
                go.Violin(
                    x=df_bonafide["model"],
                    y=df_bonafide["score"],
                    name=legend_map["bonafide"],
                    legendgroup=legend_map["bonafide"],
                    side="positive",  # Swapped to 'positive' (right) to match Matplotlib
                    line_color=color_map["bonafide"]["line"],
                    fillcolor=color_map["bonafide"]["fill"],
                    **kw,
                )
            )
            # --- Attack Trace (Left Side) ---
            df_attack = df[(df["model"] == model_name) & (df["label"] == "attack")]
            fig.add_trace(
                go.Violin(
                    x=df_attack["model"],
                    y=df_attack["score"],
                    name=legend_map["attack"],
                    legendgroup=legend_map["attack"],
                    side="negative",  # Swapped to 'negative' (left) to match Matplotlib
                    line_color=color_map["attack"]["line"],
                    fillcolor=color_map["attack"]["fill"],
                    **kw,
                )
            )

        # 4. Update traces and layout to match the target image
        fig.update_traces(width=0.8, box_line_width=1)

        fig.update_layout(
            violinmode="overlay",
            violingap=0,
            # plot_bgcolor='white',
            yaxis=dict(
                # range=[0, 1],
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
                x=0.99,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.7)",
                # bordercolor='black',
                # borderwidth=1
            ),
        )

        # # add selected_threshold as a horizontal line here
        # fig.add_hline(
        #     y=selected_threshold.value,
        #     line_color="red",
        #     line_dash="dash",
        #     annotation_text="Threshold",
        #     annotation_position="top right",
        # )
        return fig

    return (hist_plot,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
