import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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
# ---------------------------------------
def mean_embedding_norm(df, col):
    """
    col: 'LLaMA' or 'Projected'
    """
    embs = np.stack(df[col])
    norms = np.linalg.norm(embs, axis=1)
    return norms.mean()


def collect_stats(cols):
    """
    stats[prompt][corruption]['LLaMA' | 'Projected'] = list over directories
    """
    stats = {
        p: {
            c: {cols[0]: [], cols[1]: []}
            for c in CORRUPTIONS
        }
        for p in PROMPTS
    }

    numeric_dirs = sorted(
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d)) and d.isdigit()
    )

    for d in numeric_dirs:
        dpath = os.path.join(ROOT_DIR, d)

        for c in CORRUPTIONS:
            cpath = os.path.join(dpath, c)
            if not os.path.isdir(cpath):
                continue

            for p in PROMPTS:
                csv_path = os.path.join(cpath, f"{p}_embeddings.csv")
                if not os.path.exists(csv_path):
                    continue

                df = pd.read_csv(csv_path)

                stats[p][c][cols[0]].append(
                    df[cols[0]]
                )
                stats[p][c][cols[1]].append(
                    df[cols[1]]
                )

    return stats


def plot_21_subplots(stats,cols,filename,title):
    fig, axes = plt.subplots(
        nrows=len(PROMPTS),
        ncols=len(CORRUPTIONS),
        figsize=(22, 9),
        sharey=True
    )

    for i, p in enumerate(PROMPTS):
        for j, c in enumerate(CORRUPTIONS):
            ax = axes[i, j]

            llama_vals = np.array(stats[p][c][cols[0]])
            proj_vals = np.array(stats[p][c][cols[1]])
            llama_mean = np.mean(llama_vals,axis=0)
            proj_mean = np.mean(proj_vals,axis=0)

            ax.plot(llama_mean,label=cols[0])
            ax.plot(proj_mean,label=cols[1])
            ax.legend(loc='upper right')
            if i == 0:
                ax.set_title(c, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{LABELS[i]}", fontsize=10)

            ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        title,
        fontsize=14
    )
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)


# ---------------- RUN ----------------
cols = ['LLaMA','Projected']
stats = collect_stats(cols)
plot_21_subplots(stats,cols,'norms.png',"Embedding Magnitude")
cols = ['Proj–LLaMA','CLIP–Proj']
stats = collect_stats(cols)
plot_21_subplots(stats,cols,'sim.png',"Cosine Similarity ")
