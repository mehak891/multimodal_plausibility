import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
'''
def collect_logit_lens_data():
    """
    data[column][corruption][prompt] = list of DataFrames (over directories)
    """
    data = {}

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
                csv_path = os.path.join(cpath, f"{p}_logit_lens.csv")
                if not os.path.exists(csv_path):
                    continue

                df = pd.read_csv(csv_path, index_col=0)

                for col in df.columns:
                    data.setdefault(col, {})
                    data[col].setdefault(c, {})
                    data[col][c].setdefault(p, [])
                    data[col][c][p].append(df[[col]])

    return data


def plot_column_heatmaps(data):
    """
    One figure per column
    Each figure: 3 x 7 subplots
    """
    for col, col_data in data.items():
        fig, axes = plt.subplots(
            nrows=len(PROMPTS),
            ncols=len(CORRUPTIONS),
            figsize=(40, 20),
            sharey=True
        )

        for i, p in enumerate(PROMPTS):
            for j, c in enumerate(CORRUPTIONS):
                ax = axes[i, j]

                if c not in col_data or p not in col_data[c]:
                    ax.axis("off")
                    continue

                # Average over directories
                dfs = col_data[c][p]
                mean_df = np.array(dfs).mean(axis=0)
                
                im = ax.imshow(mean_df, aspect="auto", cmap="viridis")

                if i == 0:
                    ax.set_title(c, fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"{LABELS[i]}", fontsize=10)

                ax.set_xticks([])
                ax.set_yticks(range(len(mean_df)))
                ax.set_yticklabels(labels)

        fig.suptitle(
            f"Logit Lens Heatmaps for `{col}`\n"
            "Rows: Layers | Averaged over datasets",
            fontsize=14
        )

        plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        plt.tight_layout()
        plt.show()


# ---------------- RUN ----------------
logit_lens_data = collect_logit_lens_data()
plot_column_heatmaps(logit_lens_data)

'''


def load_and_average(column_name, corruption, prompt):
    """
    Returns averaged DataFrame (rows = layers, col = column_name)
    """
    dfs = []

    for d in os.listdir(ROOT_DIR):
        dpath = os.path.join(ROOT_DIR, d)
        if not (os.path.isdir(dpath) and d.isdigit()):
            continue

        csv_path = os.path.join(
            dpath, corruption, f"{prompt}_logit_lens.csv"
        )

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df[column_name])

    if len(dfs) == 0:
        return None

    return np.array(dfs).mean(axis=0)


def plot_and_save():
    # discover columns from first available file
    for d in os.listdir(ROOT_DIR):
        dpath = os.path.join(ROOT_DIR, d)
        if not (os.path.isdir(dpath) and d.isdigit()):
            continue

        for c in CORRUPTIONS:
            for p in PROMPTS:
                f = os.path.join(dpath, c, f"{p}_logit_lens.csv")
                if os.path.exists(f):
                    columns = pd.read_csv(f, nrows=1).columns.tolist()
                    break
            else:
                continue
            break
        break
    for col in columns:
        for c in CORRUPTIONS:
            fig, axes = plt.subplots(
                nrows=1,
                ncols=len(PROMPTS),
                figsize=(10, 16),
                sharey=True
            )

            for i, p in enumerate(PROMPTS):
                ax = axes[i]
                mean_df = load_and_average(col, c, p)

                if mean_df is None:
                    ax.axis("off")
                    continue

                im = ax.imshow(mean_df.reshape(-1,1), aspect="auto", cmap="viridis")

                ax.set_title(f"{LABELS[i]}")
                ax.set_xticks([])
                ax.set_yticks(range(len(mean_df)))
                ax.set_yticklabels(labels)

            fig.suptitle(
                f"Corruption: {c}",
                fontsize=14
            )

            plt.colorbar(im, ax=axes, shrink=0.6)
            #plt.tight_layout()

            save_path = os.path.join(
                SAVE_DIR, f"{col}_{c}.png"
            )
            plt.savefig(save_path, dpi=300)
            plt.close(fig)

            print(f"Saved: {save_path}")


# ---------------- RUN ----------------
plot_and_save()