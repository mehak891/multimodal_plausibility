
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()
import torch
import torch.nn.functional as F
import numpy as np
import cv2
print("Matplotlib backend:", matplotlib.get_backend())
import gc
import pandas as pd
NUM_LAYERS = 32

def save_comparison_plot(images, outputs, question, save_path):
    variants = list(images.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 5))

    for i, v in enumerate(variants):
        axes[i].imshow(images[v])
        axes[i].axis("off")
        key = list(outputs[v])[0]
        axes[i].set_title(
            f"{v}\n\n{outputs[v][key][:10]}...",
            fontsize=9
        )

    fig.suptitle(question, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def _flatten(x):
    """
    Accepts [B, N, D] or [N, D] → returns [N, D]
    """
    if x.dim() == 3:
        return x.mean(dim=0)
    return x


def plot_embedding_similarity_and_norms(
    clip_embeds,
    proj_embeds,
    llama_embeds,
    save_path,
    title=None
):
    """
    clip_embeds: CLIP vision embeddings (pre-projector)
    proj_embeds: Projected vision embeddings
    llama_embeds: LLaMA token embeddings (residual stream)
    """
    df = pd.DataFrame()
    print(f"CLIP {clip_embeds.shape} PROJECTED {proj_embeds.shape} LLAMA {llama_embeds.shape}")
    # ---- flatten ----
    clip = clip_embeds[0,1:,:]
    proj = proj_embeds[0,:,:]
    llama = llama_embeds[0,:proj.shape[0],:]
    print(f"CLIP {clip.shape} PROJECTED {proj.shape} LLAMA {llama.shape}")
    
    # ---- cosine similarities (token-wise) ----
    cos_cp = F.cosine_similarity(clip, proj[:,:1024], dim=-1)
    cos_pl = F.cosine_similarity(proj, llama, dim=-1)
    cos_cl = F.cosine_similarity(clip, llama[:,:1024], dim=-1)

    cos_values = [cos_cp, cos_pl, cos_cl]
    cos_labels = ["CLIP–Proj", "Proj–LLaMA", "CLIP–LLaMA"]
    # ---- norms ----
    norms = [
        clip.norm(dim=-1),
        proj.norm(dim=-1),
        llama.norm(dim=-1)
    ]
    norm_labels = ["CLIP", "Projected", "LLaMA"]

    # ---- plotting ----
    fig, axes = plt.subplots(1,2, figsize=(12, 6))

    for i, label in enumerate(norm_labels):
        axes[0].plot(norms[i], label=norm_labels[i])
        df[norm_labels[i]] = norms[i]
    axes[0].set_ylabel("Embedding Norm")
    axes[0].legend()
    axes[0].set_title("Per-Vision-Token Embedding Norms")

    # ---- Cosine similarity ----
    for i, label in enumerate(cos_labels):
        axes[1].plot(cos_values[i], label=cos_labels[i])
        df[cos_labels[i]] = cos_values[i]
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_xlabel("Vision token index (patch)")
    axes[1].legend()
    axes[1].set_title("Similarity Alignment")


    plt.tight_layout()
    plt.savefig(f"{save_path}_embeddings.png")
    df.to_csv(f"{save_path}_embeddings.csv",index=False)
    plt.close()
    gc.collect()



def plot_attention_overlay_no_grid(
    image,
    attn_maps,
    num_vision_tokens,
    save_path,
    title=None,
    alpha=0.4
):
    """
    image: PIL Image or numpy array (H, W, 3)
    attn_maps: list of [B, H, T, T]
    num_vision_tokens: number of vision tokens at sequence start
    """
    df = pd.DataFrame()
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    H, W = image.shape[:2]
    print(image.shape)
    fig, axes = plt.subplots(1, len(attn_maps), figsize=(4 * len(attn_maps), 6))
    if len(attn_maps) == 1:
        axes = [axes]

    for i, attn in enumerate(attn_maps):
        # mean over heads
        #print(f"Attention shape {attn.shape}")
        attn_mean = attn.mean(dim=1)[0]  # [T, T]
        #print(f"Attention mean shape {attn_mean.shape}")
        # attention from last token
        last_attn = attn_mean[-1]
        df[f"Layer_{i}"] = last_attn
        # only vision tokens
        vision_attn = last_attn[:num_vision_tokens].cpu().numpy()
        #print(f"Vision attention {vision_attn.shape}")
        vision_attn = vision_attn[1:]
        h = w = int(np.sqrt(vision_attn.shape[0]))
        
        # normalize
        vision_attn -= vision_attn.min()
        vision_attn /= vision_attn.max() + 1e-6
        # distribute uniformly over image
        heatmap = vision_attn.reshape(h, w).astype(np.float32)
        attn_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize between 0 and 1
        attn_resized -= attn_resized.min()
        attn_resized /= attn_resized.max() + 1e-6
        axes[i].imshow(image)
        im = axes[i].imshow(attn_resized, cmap="jet", alpha=alpha)
        axes[i].set_title(f"Layer {i}")
        axes[i].axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04)  # small colorbar

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    #plt.savefig(f"{save_path}_attention.png")
    df.to_csv(f"{save_path}_attention.csv",index=False)
    plt.close()

def interleave_attn_mlp_dict(attn_dict, mlp_dict):
    """
    attn_dict: {layer_idx: tensor}
    mlp_dict:  {layer_idx: tensor}

    Returns:
        interleaved: [Attn_0, MLP_0, Attn_1, MLP_1, ...]
        labels:      ["Attn 0", "MLP 0", "Attn 1", "MLP 1", ...]
    """
    interleaved = []
    labels = []

    # only layers present in BOTH
    layers = range(NUM_LAYERS)#sorted(set(attn_dict.keys()) & set(mlp_dict.keys()))
    interleaved_logits = []
    for layer in layers:
        interleaved.append(attn_dict[layer])
        labels.append(f"Attn {layer}")

        interleaved.append(mlp_dict[layer])
        labels.append(f"MLP {layer}")

    return interleaved, labels


def entropy(probs):
    probs = probs.float()
    eps = 1e-12
    return -((probs+eps) * torch.log(probs + eps)).sum().item()

def save_logit_lens_interleaved_gen_vs_gt(
    attn_out,          # dict[layer -> tensor]
    mlp_out,           # dict[layer -> tensor]
    gen_token_ids,
    gt_token_ids,
    yes_token_ids,
    no_token_ids,
    attn_logit,
    mlp_logit,
    save_path,
    title=None
):
    interleaved, labels = interleave_attn_mlp_dict(
        attn_out, mlp_out
    )

    gen_scores = []
    gt_scores = []
    yes_scores = []
    no_scores =[]
    entropy_val = []
    indices = [-1]
    for h in interleaved:
        # h shape: [B, T, V] or [T, V]
        if h.dim() == 2:
            h = h.unsqueeze(0)
        gen_scores.append(
            torch.stack([h[:, indices, tid] for tid in gen_token_ids]).mean().item()
        )
        yes_scores.append(
            torch.stack([h[:, indices, tid] for tid in yes_token_ids]).mean().item()
        )
        no_scores.append(
            torch.stack([h[:, indices, tid] for tid in no_token_ids]).mean().item()
        )
        entropy_val.append(
            entropy(h[:, indices, :])
        )
        gt_scores.append(
            torch.stack([h[:, indices, tid] for tid in gt_token_ids]).mean().item()
        )

    delta = []
    #print(f"{attn_logit[0].shape} and {mlp_logit[0].shape}")
    for i in range(NUM_LAYERS):
        delta.append(
            torch.stack([attn_logit[i][:, indices, tid] for tid in gen_token_ids]).mean().item() - torch.stack([attn_logit[i][:, indices, tid] for tid in gt_token_ids]).mean().item() 
        )
        delta.append(
            torch.stack([mlp_logit[i][:, indices, tid] for tid in gen_token_ids]).mean().item() - torch.stack([mlp_logit[i][:, indices, tid] for tid in gt_token_ids]).mean().item() 
        )
 

    #print(gen_scores)
    #print(gt_scores)
    #print(entropy_val)
    #print(delta)
    x = list(range(len(interleaved)))

    fig_height = max(6, 0.35 * len(labels))  # scale with layers
    fig, axes = plt.subplots(
        1,6,
        figsize=(7, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1,1,1,1,1], "wspace": 0.25}
    )

    im0 = axes[0].imshow(np.array(gen_scores).reshape(-1, 1), aspect="auto", cmap="viridis")
    axes[0].set_title("Generated")
    axes[0].set_xticks([])
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    #axes[0].set_ylabel("Interleaved Layers")

    # Ground truth heatmap
    im1 = axes[1].imshow(np.array(gt_scores).reshape(-1, 1), aspect="auto", cmap="viridis")
    axes[1].set_title("Answer")
    axes[1].set_xticks([])
    axes[1].set_yticks(range(len(labels)))
    #axes[1].set_yticklabels(labels)

    im2 = axes[2].imshow(np.array(entropy_val).reshape(-1, 1), aspect="auto", cmap="viridis")
    axes[2].set_title("Entropy")
    axes[2].set_xticks([])
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_yticklabels(labels)
    #axes[2].set_ylabel("Interleaved Layers")

    im3 = axes[3].imshow(np.array(delta).reshape(-1, 1), aspect="auto", cmap="viridis")
    axes[3].set_title("Delta")
    axes[3].set_xticks([])
    axes[3].set_yticks(range(len(labels)))
    axes[3].set_yticklabels(labels)
    #axes[3].set_ylabel("Interleaved Layers")
    
    im4 = axes[4].imshow(np.array(yes_scores).reshape(-1, 1), aspect="auto", cmap="viridis")
    axes[4].set_title("Yes")
    axes[4].set_xticks([])
    axes[4].set_yticks(range(len(labels)))
    axes[4].set_yticklabels(labels)
    #axes[4].set_ylabel("Interleaved Layers")
    
    im5 = axes[5].imshow(np.array(no_scores).reshape(-1, 1), aspect="auto", cmap="viridis")
    axes[5].set_title("No")
    axes[5].set_xticks([])
    axes[5].set_yticks(range(len(labels)))
    axes[5].set_yticklabels(labels)
    #axes[5].set_ylabel("Interleaved Layers")

    df = pd.DataFrame()
    df['Generated'] = gen_scores
    df['Answer'] = gt_scores
    df['Yes'] = yes_scores
    df['No'] = no_scores
    df['Delta'] = delta
    df['Entropy'] = entropy_val

    fig.colorbar(im0, ax=axes, fraction=0.04)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{save_path}_logit_lens.png", dpi=200)
    df.to_csv(f"{save_path}_logit_lens.csv",index=False)
    plt.close()
