import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# ---------------- CONFIG ----------------
ROOT_DIR = "./outputs/all_generations"

CORRUPTIONS = [
    "original",
    "no_visual",
    "light_blur",
    "medium_blur",
    "heavy_blur",
    "slight_noise",
    "noise"
]

PROMPTS = [0, 1, 2]  # corresponds to 0_embeddings.csv, 1_embeddings.csv, 2_embeddings.csv
LABELS = ['DEFAULT','UNCERTAINITY','ABSTENTION']
labels = [[f'attn_{i}',f'mlp_{i}'] for i in range(32)]
labels = [item for sublist in labels for item in sublist]
SAVE_DIR = './outputs/plots'

def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def entropy(vals, bins=100):
    hist, _ = np.histogram(vals, bins=bins, density=True)
    hist += 1e-8
    hist /= hist.sum()
    return -(hist * np.log(hist)).sum()

def plot_kde(ax, vals, color):
    vals = vals[~np.isnan(vals)]
    if len(vals) < 10:
        return

    kde = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 300)
    ys = kde(xs)

    ax.plot(xs, ys, color=color, lw=1.8)
    ax.fill_between(xs, ys, color=color, alpha=0.25)

def load_attention_values(corruption, prompt):
    layer_vals = {}

    for d in os.listdir(ROOT_DIR):
        dpath = os.path.join(ROOT_DIR, d)
        if not (os.path.isdir(dpath) and d.isdigit()):
            continue

        path = os.path.join(dpath, corruption, f"{prompt}_attention.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        for col in df.columns:
            layer_vals.setdefault(col, []).append(df[col].values)

    for k in layer_vals:
        layer_vals[k] = np.concatenate(layer_vals[k])

    return layer_vals


def plot_attention_density(corruption):
    fig = plt.figure(figsize=(21, 7))
    outer = fig.add_gridspec(1, 3, wspace=0.18)

    for i, p in enumerate(PROMPTS):
        layer_vals = load_attention_values(corruption, p)
        layers = list(layer_vals.keys())
        n_layers = len(layers)

        ncols = 4
        nrows = int(np.ceil(n_layers / ncols))
        inner = outer[i].subgridspec(nrows, ncols, hspace=0.65, wspace=0.45)

        for idx, layer in enumerate(layers):
            r, c = divmod(idx, ncols)
            ax = fig.add_subplot(inner[r, c])

            vals = layer_vals[layer]
            H = entropy(vals)

            plot_kde(ax, vals, color="tab:blue")

            ax.set_title(f"{layer}\nH={H:.2f}", fontsize=8)
            clean_axis(ax)

        # hide unused cells
        for idx in range(len(layers), nrows * ncols):
            fig.add_subplot(inner[idx // ncols, idx % ncols]).axis("off")
            row = idx // ncols
            col = idx % ncols

            # Show x-axis ONLY on bottom row
            if row == nrows - 1:
                ax.tick_params(axis="x", labelsize=8)
                ax.set_xlabel("Attention", fontsize=8)
            else:
                ax.set_xticks([])

            # Never show y-axis for density plots
            ax.set_yticks([])

            # Light grid only horizontally
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            xmin, xmax = 0.0, 1.0  # or np.percentile(vals, [1, 99])
            #ax.set_xlim(xmin, xmax)
        fig.text(
            0.17 + i * 0.28,
            0.93,
            f"Prompt {p}",
            ha="center",
            fontsize=13,
            weight="bold"
        )
        shared_ax = fig.add_subplot(outer[i])
        shared_ax.set_xlabel("Attention weight", fontsize=10)
        #shared_ax.tick_params(axis="x", labelsize=9)
        shared_ax.set_yticks([])
        shared_ax.patch.set_alpha(0)
        for spine in shared_ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"Layer-wise Attention Density Across Tokens\nCorruption: {corruption}",
        fontsize=16,
        y=0.99
    )

    save_path = os.path.join(
        SAVE_DIR, f"attention_density_{corruption}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")

for c in CORRUPTIONS:
    plot_attention_density(c)