import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT_DIR = "./outputs/all_generations"

PROMPTS = ["default", "evidence_only", "abstention_allowed"]
BLUR_LEVELS = [
    "original",
    "no_visual",
    "light_blur",
    "medium_blur",
    "heavy_blur",
    "slight_noise",
    "noise"
]

ABSTAIN_TOKEN = ["","insufficient evidence"]

def load_all_qa(root_dir):
    qa_list = []
    for d in sorted(os.listdir(root_dir)):
        qa_path = os.path.join(root_dir, d, "qa.json")
        if os.path.exists(qa_path):
            with open(qa_path, "r") as f:
                qa_list.append(json.load(f))
    return qa_list

def compute_hallucination_percentages(qa_list):
    """
    Returns:
        halluc[prompt][blur] = percentage hallucinated
    """

    counts = defaultdict(lambda: defaultdict(lambda: {"halluc": 0, "total": 0}))

    for qa in qa_list:
        gt_answer = qa["answer"].lower()
        gt_truth = gt_answer
        outputs = qa["outputs"]

        for blur in BLUR_LEVELS:
            if blur not in outputs:
                continue

            for i,prompt in enumerate(PROMPTS):
                #if prompt not in list(outputs[blur].values()):
                #    continue

                pred = list(outputs[blur].values())[i].lower()
                counts[prompt][blur]["total"] += 1

                # =====================
                # UNANSWERABLE
                # =====================
                if blur in ["no_visual","medium_blur","heavy_blur","noise"]:
                    if pred not in ABSTAIN_TOKEN:
                        counts[prompt][blur]["halluc"] += 1

                # =====================
                # ANSWERABLE
                # =====================
                else:
                    if gt_answer not in pred:
                        counts[prompt][blur]["halluc"] += 1

    halluc = defaultdict(dict)

    for prompt in PROMPTS:
        for blur in BLUR_LEVELS:
            c = counts[prompt][blur]
            halluc[prompt][blur] = (
                100.0 * c["halluc"] / c["total"]
                if c["total"] > 0 else 0.0
            )

    return halluc


def plot_hallucination_bars(halluc, save_path="hallucination_bars.png"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for idx, prompt in enumerate(PROMPTS):
        ax = axes[idx]

        values = [halluc[prompt][b] for b in BLUR_LEVELS]
        ax.bar(BLUR_LEVELS, values)

        ax.set_title(prompt.replace("_", " ").title())
        ax.set_xticklabels(BLUR_LEVELS, rotation=90)

        ax.grid(axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Hallucinated Answers (%)")

    fig.supxlabel("Corruption Level (Blur / Noise)")
    fig.suptitle("Hallucination Rate Across Prompts and Evidence Degradation")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_hallucination_by_corruption(halluc, save_path="hallucination_by_corruption.png"):
    """
    halluc[prompt][blur] = percentage hallucinated
    """

    fig, axes = plt.subplots(1, 7, figsize=(16, 4), sharey=True)

    for idx, blur in enumerate(BLUR_LEVELS):
        ax = axes[idx]

        values = [halluc[prompt][blur] for prompt in PROMPTS]
        ax.bar(PROMPTS, values)

        ax.set_title(blur.replace("_", " ").title())
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

        ax.set_xticklabels(
            [p.replace("_", " ").title() for p in PROMPTS],
            rotation=90
        )

        if idx == 0:
            ax.set_ylabel("Hallucinated Answers (%)")

    fig.supxlabel("Prompt Condition")
    fig.suptitle("Hallucination Rate by Prompt Across Corruption Levels")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    qa_list = load_all_qa(ROOT_DIR)
    halluc = compute_hallucination_percentages(qa_list)
    plot_hallucination_bars(halluc)
    plot_hallucination_by_corruption(halluc)


    print("3-panel hallucination bar plot saved.")

if __name__ == "__main__":
    main()
