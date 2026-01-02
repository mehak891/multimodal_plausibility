import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def kl_divergence(p, q):
    eps = 1e-8
    p = p + eps
    q = q + eps
    return np.sum(p * np.log(p / q))

def attention_distribution(corruption, prompt, layer, bins):
    vals = []

    for d in os.listdir(ROOT_DIR):
        dpath = os.path.join(ROOT_DIR, d)
        if not (os.path.isdir(dpath) and d.isdigit()):
            continue

        path = os.path.join(
            dpath, corruption, f"{prompt}_attention.csv"
        )
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        vals.append(df[layer].values)

    vals = np.concatenate(vals)
    hist, _ = np.histogram(vals, bins=bins, density=True)
    hist = hist / hist.sum()
    return hist

def plot_kl_for_corruption(corruption):
    # discover layers
    for d in os.listdir(ROOT_DIR):
        dpath = os.path.join(ROOT_DIR, d)
        if not (os.path.isdir(dpath) and d.isdigit()):
            continue

        f = os.path.join(dpath, "original", "0_attention.csv")
        if os.path.exists(f):
            layers = pd.read_csv(f, nrows=1).columns.tolist()
            break

    bins = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i,p in enumerate(PROMPTS):
        kl_vals = []

        for layer in layers:
            p_orig = attention_distribution("original", p, layer, bins)
            p_corr = attention_distribution(corruption, p, layer, bins)

            kl_vals.append(kl_divergence(p_orig, p_corr))

        ax.plot(
            range(len(layers)),
            kl_vals,
            marker="o",
            label=f"{LABELS[i]}"
        )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("KL divergence")
    ax.set_title(f"Attention Drift under {corruption}")
    ax.legend()
    ax.grid(alpha=0.3)
    save_path = os.path.join(
        SAVE_DIR, f"attention_kl_{corruption}.png"
    )
    plt.tight_layout()
    plt.savefig(save_path)

for c in CORRUPTIONS:
    if c != "original":
        plot_kl_for_corruption(c)