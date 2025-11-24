import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parent
RUN_CONFIGS = [
    {"file": "resultsAB.csv", "train": "A", "test": "B"},
    {"file": "resultsBA.csv", "train": "B", "test": "A"},
]
COLUMNS = [
    "learning rate",
    "epochs",
    "perceptron number",
    "random seed",
    "correct matches",
    "dataset size",
    "success rate",
]
HYPER_COLS = ["learning rate", "epochs", "perceptron number", "random seed"]

sns.set_theme(style="whitegrid", context="talk")


def load_runs():
    frames = []
    for cfg in RUN_CONFIGS:
        path = DATA_DIR / cfg["file"]
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        frame = pd.read_csv(path, names=COLUMNS, skiprows=1)
        frame["success rate"] = frame["success rate"].str.rstrip("%" ).astype(float)
        frame["train_set"] = cfg["train"]
        frame["test_set"] = cfg["test"]
        frame["scenario"] = f"{cfg['train']}→{cfg['test']}"
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No result CSVs were found. Check RUN_CONFIGS paths.")
    return pd.concat(frames, ignore_index=True)


def describe_runs(df):
    print("--- Combined Runs Overview ---")
    print(df.head())
    print("Run counts by scenario:\n", df["scenario"].value_counts())
    best = df.sort_values("success rate", ascending=False).groupby("scenario").head(5)
    print("\nTop 5 configs per scenario:")
    print(best[HYPER_COLS + ["success rate", "scenario"]])


def plot_success_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df,
        x="scenario",
        y="success rate",
        hue="scenario",
        estimator="median",
        errorbar="sd",
        palette="viridis",
        legend=False,
    )
    sns.stripplot(data=df, x="scenario", y="success rate", color="black", alpha=0.5)
    plt.ylim(80, 100) 
    plt.title("Success Rate Distribution by Train/Test Split")
    plt.ylabel("Success Rate (%)")
    plt.show()


def plot_param_relationships(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, param in zip(axes, HYPER_COLS):
        sns.scatterplot(data=df, x=param, y="success rate", hue="scenario", ax=ax, style="scenario", s=120)
        sns.lineplot(
            data=df,
            x=param,
            y="success rate",
            hue="scenario",
            ax=ax,
            estimator="median",
            lw=2,
            legend=False,
        )
        ax.set_title(f"Success vs {param.title()}")
        ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(df["scenario"].unique()))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_epoch_perceptron_heatmap(df):
    scenarios = df["scenario"].unique()
    ncols = len(scenarios)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        subset = df[df["scenario"] == scenario]
        pivot = subset.pivot_table(
            index="perceptron number", columns="epochs", values="success rate", aggfunc="mean"
        )
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title(f"{scenario}: Mean Success")
        ax.set_ylabel("Perceptrons")
        ax.set_xlabel("Epochs")
    plt.tight_layout()
    plt.show()


def plot_generalization_gap(df):
    pivot = df.pivot_table(index=HYPER_COLS, columns="scenario", values="success rate")
    scenarios = list(pivot.columns)
    if len(scenarios) < 2:
        print("Not enough scenarios to compute generalization gap.")
        return
    gap_pairs = []
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            gap = pivot[scenarios[i]] - pivot[scenarios[j]]
            pair_df = gap.reset_index()
            pair_df.columns = HYPER_COLS + ["gap"]
            pair_df["pair"] = f"{scenarios[i]} - {scenarios[j]}"
            gap_pairs.append(pair_df)
    gap_df = pd.concat(gap_pairs, ignore_index=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=gap_df, x="perceptron number", y="gap", hue="pair", errorbar=None)
    plt.axhline(0, color="black", lw=1)
    plt.title("Generalization Gap Across Train/Test Directions")
    plt.ylabel("Success Difference (pp)")
    plt.show()


def main():
    df = load_runs()
    describe_runs(df)
    plot_success_distribution(df)
    plot_param_relationships(df)
    plot_epoch_perceptron_heatmap(df)
    plot_generalization_gap(df)


if __name__ == "__main__":
    main()