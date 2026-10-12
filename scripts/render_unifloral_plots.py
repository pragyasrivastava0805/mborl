#!/usr/bin/env python3
"""Usage: python3 render_unifloral_plots.py <repo_dir> <results_dir> <out_pdf> <out_png>

Renders the 4x5 D4RL Unifloral UCB bandit policy selection curves (20 datasets x 12 methods)
exactly as specified in evaluation_plots.ipynb and launcher.md Section 6, and saves both PDF and PNG.
"""
import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    repo_dir = os.path.expanduser(sys.argv[1])
    results_dir = os.path.expanduser(sys.argv[2])
    out_pdf = os.path.expanduser(sys.argv[3])
    out_png = os.path.expanduser(sys.argv[4])

    sys.path.insert(0, repo_dir)
    import evaluation

    # --- Plot style configuration from evaluation_plots.ipynb ---
    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["figure.figsize"] = [12, 10]
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    df = evaluation.load_results_dataframe(results_dir=results_dir)
    assert len(df) == 2400

    # --- Define bandit evaluation and plotting ---
    def plot_trial(final_scores, num_subsample=8, **kwargs):
        runs = np.stack(final_scores.values)
        res = evaluation.bootstrap_bandit_trials(
            runs, num_subsample=num_subsample, n_bootstraps=2000, num_repeats=500
        )
        line = plt.semilogx(res["pulls"], res["estimated_bests_mean"], **kwargs)
        plt.fill_between(
            res["pulls"],
            res["estimated_bests_ci_low"],
            res["estimated_bests_ci_high"],
            alpha=0.3,
            color=line[0].get_color(),
        )

    plot_df = df.copy()
    _algos = plot_df["algorithm"].unique()
    _palette = sns.color_palette("colorblind" if len(_algos) <= 10 else "tab20", len(_algos))
    ALGORITHM_COLOR_DICT = dict(zip(_algos, _palette))
    plot_df["dataset"] = pd.Categorical(plot_df["dataset"])
    num_datasets = len(plot_df["dataset"].unique())
    num_algorithms = len(plot_df["algorithm"].unique())
    g = sns.FacetGrid(
        plot_df,
        col="dataset",
        hue="algorithm",
        col_wrap=math.ceil(np.sqrt(num_datasets)),
        sharex=False,
        sharey=False,
        palette=ALGORITHM_COLOR_DICT,
    )
    print("Running bandit evaluation... ", end="", flush=True)
    g.map(plot_trial, "final_scores")
    print("Done.", flush=True)

    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(ax.get_title().replace("dataset = ", ""))

    fig = plt.gcf()
    big_ax = fig.add_subplot(111, frameon=False)
    big_ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    big_ax.grid(False)
    plt.xlabel("Number of policy evaluations")
    plt.ylabel("Mean score")

    g.add_legend(title="Algorithm")
    leg = g._legend
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        line.set_linewidth(5.0)
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 0.0),
        ncol=1 + (num_algorithms // 2),
        frameon=True,
        edgecolor="gray",
        fancybox=True,
        shadow=False,
    )

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
