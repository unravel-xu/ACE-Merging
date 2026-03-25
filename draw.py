import torch
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
import pandas as pd
from scipy.stats import norm, shapiro
import re  # 导入正则表达式模块
from utils.task_vector import TaskVector
from utils.common import construct_task_vectors_list

# ====== 配置区 ======
config_save_path = "./data/configs/vit"
task_vector_dir = "./data/task_vectors/"

# model_type = 'roberta-large'  # ← 可改, e.g., 'gpt2', 'ViT-B-16'
# model_type = "ViT-L-14"
# model_type = "ViT-B-16"
model_type = "gpt"

model_id_list = [
    # f'{model_type}',
    # f'./data/models/lm/checkpoints/{model_type}/cola/{model_type}_lr1e-05/',
    # f'./data/models/lm/checkpoints/{model_type}/sst2/{model_type}_lr1e-05/',
    # f'./data/models/lm/checkpoints/{model_type}/mrpc/{model_type}_lr1e-05/',
    # f'./data/models/lm/checkpoints/{model_type}/stsb/{model_type}_lr1e-05/',
    # f'./data/models/lm/checkpoints/{model_type}/qqp/{model_type}_lr1e-05/',
    # f'./data/models/lm/checkpoints/{model_type}/qnli/{model_type}_lr1e-05/',
    # f'./data/models/lm/checkpoints/{model_type}/mnli/{model_type}_lr1e-05/',
    # f'./data/models/lm/checkpoints/{model_type}/rte/{model_type}_lr1e-05/',
    # f"./data/models/vit/checkpoints/{model_type}/MNISTVal/nonlinear_zeroshot.pt",
    # f"./data/models/vit/checkpoints/{model_type}/MNISTVal/nonlinear_finetuned.pt",
    # f"./data/models/vit/checkpoints/{model_type}/CarsVal/nonlinear_finetuned.pt",
    # f"./data/models/vit/checkpoints/{model_type}/DTDVal/nonlinear_finetuned.pt",
    # f"./data/models/vit/checkpoints/{model_type}/EuroSATVal/nonlinear_finetuned.pt",
    # f"./data/models/vit/checkpoints/{model_type}/GTSRBVal/nonlinear_finetuned.pt",
    # f"./data/models/vit/checkpoints/{model_type}/RESISC45Val/nonlinear_finetuned.pt",
    # f"./data/models/vit/checkpoints/{model_type}/SUN397Val/nonlinear_finetuned.pt",
    # f"./data/models/vit/checkpoints/{model_type}/SVHNVal/nonlinear_finetuned.pt",
    # f'./data/models/vit/checkpoints/{model_type}/PCAMVal/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/CIFAR100Val/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/STL10Val/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/OxfordIIITPetVal/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/Flowers102Val/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/FER2013Val/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/CIFAR10Val/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/Food101Val/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/RenderedSST2Val/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/EMNISTVal/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/FashionMNISTVal/nonlinear_finetuned.pt',
    # f'./data/models/vit/checkpoints/{model_type}/KMNISTVal/nonlinear_finetuned.pt',
    'gpt2',
    'tanganke/gpt2_cola',
    'tanganke/gpt2_sst2',
    'tanganke/gpt2_mrpc',
    'tanganke/gpt2_qqp',
    'tanganke/gpt2_mnli',
    'tanganke/gpt2_qnli',
    'tanganke/gpt2_rte'
]

task_nums = 20
base_model_id = model_id_list[0]
other_model_id_list = model_id_list[1:task_nums + 1]


# ====== 工具函数 ======
def infer_model_architecture(model_type: str) -> str:
    """根据 model_type 字符串推断架构类别"""
    mt = model_type.lower()
    if any(k in mt for k in ['vit', 'vision', 'clip']):
        return 'vit'
    elif 'roberta' in mt:
        return 'roberta'
    elif 'bert' in mt:
        return 'bert'
    elif 'gpt' in mt:
        return 'gpt'
    else:
        return 'unknown'

# ==================== 新增过滤逻辑函数 ====================
def get_exclude_patterns(arch: str) -> list:
    """
    根据模型架构返回要排除的参数名称的正则表达式列表。
    """
    # 通用规则 (来自 'llm' 配置)
    # 使用 ^ 和 $ 来确保 lm_head.weight 是精确匹配
    patterns = [".*embed_tokens.*", "^lm_head\.weight$"]

    if arch == 'gpt':
        # 对应 'gpt2' 的规则
        patterns.extend([".*score.*", ".*wpe.*"])
        return patterns
    elif arch in ['bert', 'roberta']:
        # 对应 'bert' 的规则
        # 注意: 这里的 .*embeddings.* 会覆盖通用规则的 .*embed_tokens.*
        return [".*classifier.*", ".*bias.*", ".*LayerNorm.*", ".*embeddings.*"]
    elif arch == 'vit':
        # 对应 'vit' 的规则，没有额外的 regex 过滤
        return []
    
    # 对于其他未明确指定的 LM 架构，返回通用规则
    return patterns

def should_exclude_by_regex(layer_name: str, patterns: list) -> bool:
    """
    检查 layer_name 是否匹配任何一个排除模式。
    """
    if not patterns:
        return False
    for pattern in patterns:
        if re.search(pattern, layer_name):
            return True
    return False
# ========================================================


def build_layer_type_map(layer_names, arch: str):
    layer_type_map = {}
    if arch in ['vit', 'clip']:
        for name in layer_names:
            lname = name.lower()
            if "positional_embedding" in lname or "token_embedding" in lname:
                layer_type_map[name] = "Embedding"
            # elif "text_projection" in lname:
            #     layer_type_map[name] = "Text-Proj"
            elif "attn.in_proj_weight" in lname:
                layer_type_map[name] = "Attn-In"
            elif "attn.out_proj.weight" in lname:
                layer_type_map[name] = "Attn-Out"
            elif "mlp.c_fc.weight" in lname:
                layer_type_map[name] = "MLP-Fc"
            elif "mlp.c_proj.weight" in lname:
                layer_type_map[name] = "MLP-Proj"
            # elif "visual.proj" in lname:
            #     layer_type_map[name] = "Visual-Proj"
            # else:
            #     layer_type_map[name] = "Other"
    
    elif arch in ['bert', 'roberta']:
        for name in layer_names:
            lname = name.lower()
            if "attention.self.query" in lname:
                layer_type_map[name] = "Attn-Q"
            elif "attention.self.key" in lname:
                layer_type_map[name] = "Attn-K"
            elif "attention.self.value" in lname:
                layer_type_map[name] = "Attn-V"
            elif "attention.output.dense" in lname:
                layer_type_map[name] = "Attn-Out"
            elif "intermediate.dense" in lname:
                layer_type_map[name] = "FFN-Up"
            elif "output.dense" in lname and "attention.output" not in lname:
                layer_type_map[name] = "FFN-Down"
            # else:
            #     layer_type_map[name] = "Other"
    
    elif arch == 'gpt':
        for name in layer_names:
            lname = name.lower()
            # if "wte" in lname or "wpe" in lname:
            #     layer_type_map[name] = "Embedding"
            # elif "ln_" in lname or "layernorm" in lname:
            #     layer_type_map[name] = "LayerNorm"
            if "attn.c_attn" in lname:
                layer_type_map[name] = "Attn-QKV"
            elif "mlp.c_fc" in lname:
                layer_type_map[name] = "FFN-Up"
            elif "mlp.c_proj" in lname:
                layer_type_map[name] = "FFN-Down"
            elif "attn.c_proj" in lname:
                layer_type_map[name] = "Attn-Proj"
            # else:
            #     layer_type_map[name] = "Other"
    
    else:
        for name in layer_names:
            layer_type_map[name] = "Other"
    
    return layer_type_map

def get_order_and_shortmap(arch: str):
    if arch in ['vit', 'clip']:
        # order_full = ["Embedding", "Attn-In", "Attn-Out", "MLP-Fc", "MLP-Proj", "Text_Proj", "Visual_Proj", "Other"]
        # short_map = {"Embedding": "Emb", "Attn-In": "QKV", "Attn-Out": "A-Out", "MLP-Fc": "Fc", "MLP-Proj": "Proj", 
        #     "Text_Proj": "T-Proj", "Visual_Proj": "V-Proj", "Other": "Oth"}
        order_full = ["Embedding", "Attn-In", "Attn-Out", "MLP-Fc", "MLP-Proj"]
        short_map = {"Embedding": "Emb", "Attn-In": "QKV", "Attn-Out": "A-Out", "MLP-Fc": "Fc", "MLP-Proj": "Proj"}
    elif arch in ['bert', 'roberta']:
        # order_full = ["Attn-Q", "Attn-K", "Attn-V",
        #               "Attn-Out", "FFN-Up", "FFN-Down", "Other"]
        # short_map = {"Attn-Q": "Q", "Attn-K": "K", "Attn-V": "V", "Attn-Out": "Out", "FFN-Up": "Up", 
        #              "FFN-Down": "Down", "Other": "Oth"}
        order_full = ["Attn-Q", "Attn-K", "Attn-V",
                      "Attn-Out", "FFN-Up", "FFN-Down"]
        short_map = {"Attn-Q": "Q", "Attn-K": "K", "Attn-V": "V", "Attn-Out": "Out", "FFN-Up": "Up", 
                     "FFN-Down": "Down"}
    elif arch == 'gpt':
        order_full = ["Attn-QKV", "FFN-Up", "FFN-Down", "Attn-Proj"]
        short_map = {"Attn-QKV": "QKV", "FFN-Up": "Up", "FFN-Down": "Down", "Attn-Proj": "Proj",}
    else:
        order_full = ["Other"]
        short_map = {"Other": "Oth"}
    
    return order_full, short_map

# def visualize_structural_heterogeneity_cvpr(layer_names, gamma_values, layer_type_map, 
#                                             model_type: str, threshold=0.3):
#     """
#     Final fixed and polished version for CVPR figure.
#     Supports ViT/BERT/GPT model type visualization.
#     """
#     arch = infer_model_architecture(model_type)
#     order_full, short_map = get_order_and_shortmap(arch)
#     # ====== Theme setup ======
#     sns.set_theme(style="whitegrid", font_scale=0.95, rc={
#         "axes.edgecolor": "0.2",
#         "axes.linewidth": 0.9,
#         "font.family": "serif",
#         "font.serif": ["Times New Roman"],
#         "mathtext.fontset": "stix",
#         "xtick.labelsize": 7,
#         "ytick.labelsize": 7,
#         "axes.labelsize": 8,
#         "legend.fontsize": 8,
#         "figure.titlesize": 12,
#         "axes.titlesize": 10,
#     })

#     # ====== Layer type short name mapping ======
#     full_types = [layer_type_map.get(name, "Other") for name in layer_names]
#     short_types = [short_map.get(t, t[:4]) for t in full_types]
#     order_short = [short_map[t] for t in order_full]

#     df = pd.DataFrame({
#         "Layer": layer_names,
#         "Gamma": gamma_values,
#         "TypeFull": full_types,
#         "TypeShort": short_types
#     })
#     df["TypeShort"] = pd.Categorical(df["TypeShort"], categories=order_short, ordered=True)

#     # ====== Compute Y range ======
#     gamma_max = np.max(gamma_values) if len(gamma_values) > 0 else 0.1
#     y_max = gamma_max * 1.15

#     # ====== Figure layout ======
#     fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=False)
#     plt.subplots_adjust(wspace=0.28, bottom=0.22, top=0.82)

#     # ====== (a) Violin plot ======
#     ax1 = axes[0]
#     palette = sns.color_palette("Set2", n_colors=len(order_full))

#     sns.violinplot(
#         data=df, x="TypeShort", y="Gamma", ax=ax1,
#         order=order_short,
#         inner="box", linewidth=0.8,
#         palette=palette,
#         cut=0,
#         scale="width"
#     )

#     ax1.axhline(y=threshold, color="black", linestyle="--", lw=1.0, zorder=0)
#     ax1.set_xlabel("Layer Type", fontsize=8, labelpad=2)
#     ax1.set_ylabel(r"Heterogeneity Metric ($\gamma$)", fontsize=8, labelpad=2)
#     ax1.set_title("(a) Distribution of Heterogeneity Metric ($\\gamma$) Across Layer Types",
#                   fontsize=9, pad=5)
#     ax1.set_ylim(0, y_max)
#     ax1.grid(True, ls="--", lw=0.4, alpha=0.5, zorder=0)
#     ax1.tick_params(axis='y', which='both', left=True, right=False)

#     # 手动图例
#     legend_elements = [Patch(facecolor=palette[i], edgecolor='none', label=order_full[i])
#                        for i in range(len(order_full))]
#     ax1.legend(
#         handles=legend_elements,
#         title="Layer Types",
#         title_fontsize=6,
#         fontsize=5,
#         loc='upper right',
#         bbox_to_anchor=(1.0, 1.0),
#         frameon=True,
#         fancybox=False,
#         shadow=False,
#         framealpha=0.95,
#         borderpad=0.3,
#         handletextpad=0.4,
#         columnspacing=0.3
#     )

#     # ====== (b) Bar chart ======
#     ax2 = axes[1]
#     x = np.arange(len(layer_names))
#     gamma_arr = np.array(gamma_values)

#     # 🎨 优雅配色：低γ浅蓝紫，高γ橙色（非警示感）
#     base_color = "#8E8BFE"
#     highlight_color = "#ff7979"
#     colors = [highlight_color if g > threshold else base_color for g in gamma_arr]

#     ax2.bar(x, gamma_arr, color=colors, width=0.8, alpha=0.9, edgecolor="none", zorder=3)
#     ax2.axhline(y=threshold, color="black", linestyle="--", lw=1.0,
#                 label=fr"$\gamma = {threshold}$", zorder=1)
#     ax2.set_xlabel("Layer Index", fontsize=8, labelpad=2)
#     ax2.set_title("(b) Layer-wise Heterogeneity Metric ($\\gamma$) Profile", fontsize=9, pad=5)
#     ax2.set_ylim(0, y_max)
#     ax2.grid(True, axis="both", ls="--", lw=0.4, alpha=0.5, zorder=0)
#     ax2.set_ylabel("")
#     ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

#     step = max(1, len(layer_names) // 8)
#     xticks = np.arange(0, len(layer_names), step)
#     ax2.set_xticks(xticks)
#     ax2.set_xticklabels([str(i) for i in xticks], fontsize=7)
#     ax2.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.0), borderpad=0.3)

#     # ====== Border & Export ======
#     for ax in axes:
#         for spine in ax.spines.values():
#             spine.set_linewidth(0.2)
#             spine.set_color("0.8")
#         ax.set_facecolor('white')

#     plt.tight_layout(rect=[0, 0.03, 1, 0.9])

#     safe_name = model_type.replace("/", "_").replace("-", "_")
#     plt.savefig(f"task_heterogeneity_{safe_name}_{task_nums}.pdf", bbox_inches="tight")
#     plt.savefig(f"task_heterogeneity_{safe_name}_{task_nums}.png", bbox_inches="tight", dpi=400)
#     print(f"✅ Saved to task_heterogeneity_{safe_name}_{task_nums}.pdf/png")

def visualize_structural_heterogeneity_cvpr(layer_names, gamma_values, layer_type_map, 
                                            model_type: str, threshold=0.3):
    """
    CVPR-optimized version: narrower width (≈7.2in), taller height (4.6in),
    ideal for single-column figures with improved readability.
    """
    arch = infer_model_architecture(model_type)
    order_full, short_map = get_order_and_shortmap(arch)
    
    # ====== Theme setup: slightly larger font for readability ======
    sns.set_theme(style="whitegrid", font_scale=1.0, rc={
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.9,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.titlesize": 10.5,
    })

    # ====== Layer type short name mapping ======
    full_types = [layer_type_map.get(name, "Other") for name in layer_names]
    short_types = [short_map.get(t, t[:4]) for t in full_types]
    order_short = [short_map[t] for t in order_full]

    df = pd.DataFrame({
        "Layer": layer_names,
        "Gamma": gamma_values,
        "TypeFull": full_types,
        "TypeShort": short_types
    })
    df["TypeShort"] = pd.Categorical(df["TypeShort"], categories=order_short, ordered=True)

    # ====== Compute Y range ======
    gamma_max = np.max(gamma_values) if len(gamma_values) > 0 else 0.1
    y_max = gamma_max * 1.15

    # ====== Figure layout: narrower & taller (CVPR single-column friendly) ======
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=False)
    plt.subplots_adjust(
        wspace=0.30,      # 子图间距稍增，避免 xlabel 重叠
        bottom=0.15,      # 底部留白↓（原0.22）
        top=0.88,         # 顶部留白↑（标题/图例空间更足）
        left=0.10,        # 左边距↑（y-label 不被裁剪）
        right=0.96        # 右边距↓（紧凑但留 legend 空间）
    )

    # ====== (a) Violin plot ======
    ax1 = axes[0]
    palette = sns.color_palette("Set2", n_colors=len(order_full))

    sns.violinplot(
        data=df, x="TypeShort", y="Gamma", ax=ax1,
        order=order_short,
        inner="box", linewidth=0.8,
        palette=palette,
        cut=0,
        scale="width"
    )

    ax1.axhline(y=threshold, color="black", linestyle="--", lw=1.0, zorder=0)
    ax1.set_xlabel("Layer Type", fontsize=9, labelpad=3)
    ax1.set_ylabel(r"Heterogeneity Metric ($\gamma$)", fontsize=9, labelpad=3)
    ax1.set_title("(a) Distribution Across Layer Types", fontsize=10.5, pad=6)
    ax1.set_ylim(0, y_max)
    ax1.grid(True, ls="--", lw=0.4, alpha=0.5, zorder=0)
    ax1.tick_params(axis='y', which='both', left=True, right=False)

    # ====== (a) Violin plot 的图例移动到左侧 ======
    legend_elements = [
        Patch(facecolor=palette[i], edgecolor='none', label=order_full[i])
        for i in range(len(order_full))
    ]
    ax1.legend(
        handles=legend_elements,
        title="Layer Types",
        title_fontsize=8,
        fontsize=8,
        loc='upper left',              # ← 改为左上角
        bbox_to_anchor=(0.05, 0.97),   # ← 微调位置：距离左边界5%，顶部97%
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        borderpad=0.3,
        handletextpad=0.4,
        columnspacing=0.5,
        ncol=1                       # 单列排列，更紧凑
    )

    # ====== (b) Bar chart ======
    ax2 = axes[1]
    x = np.arange(len(layer_names))
    gamma_arr = np.array(gamma_values)

    # 🎨 配色保持您的偏好
    base_color = "#8E8BFE"
    highlight_color = "#ff7979"
    colors = [highlight_color if g > threshold else base_color for g in gamma_arr]

    ax2.bar(x, gamma_arr, color=colors, width=0.8, alpha=0.9, edgecolor="none", zorder=3)
    ax2.axhline(y=threshold, color="black", linestyle="--", lw=1.0,
                label=fr"$\gamma = {threshold:.1f}$", zorder=1)  # ← 保留一位小数！
    ax2.set_xlabel("Layer Index", fontsize=9, labelpad=3)
    ax2.set_title("(b) Layer-wise Profile", fontsize=10.5, pad=6)
    ax2.set_ylim(0, y_max)
    ax2.grid(True, axis="both", ls="--", lw=0.4, alpha=0.5, zorder=0)
    ax2.set_ylabel("")
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # X-ticks: 保证清晰（适配新高度）
    step = max(1, len(layer_names) // 10)  # 更密一点（因高度↑可容纳更多）
    xticks = np.arange(0, len(layer_names), step)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([str(i) for i in xticks], fontsize=8, rotation=0)

    ax2.legend(
        fontsize=8,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.97),
        borderpad=0.3,
        framealpha=0.95
    )

    # ====== Border & Export ======
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.2)
            spine.set_color("0.7")
        ax.set_facecolor('white')

    # tight_layout with refined padding
    plt.tight_layout(rect=[0.02, 0.08, 0.99, 0.94])

    safe_name = model_type.replace("/", "_").replace("-", "_")
    plt.savefig(f"task_heterogeneity_{safe_name}_{task_nums}.pdf", 
                bbox_inches="tight", pad_inches=0.05)
    plt.savefig(f"task_heterogeneity_{safe_name}_{task_nums}.png", 
                bbox_inches="tight", dpi=500, pad_inches=0.05)
    print(f"✅ Saved (7.2×4.6 in) to task_heterogeneity_{safe_name}_{task_nums}.pdf/png")

def visualize_weight_distribution(
    W_merged, 
    layer_name="Layer", 
    figsize=(8, 5),
    bins=100,
    show_gaussian_fit=True,
    save_fig=True
):
    # Step 1: 展平权重矩阵
    weights = W_merged.view(-1).cpu().numpy()  # [N]
    
    # Step 2: 计算统计量
    mean_val = np.mean(weights)
    std_val = np.std(weights)
    min_val = np.min(weights)
    max_val = np.max(weights)
    
    # Step 3: 绘制直方图
    sns.set_theme(style="whitegrid", font_scale=1.0, rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 直方图
    counts, bins_edges, patches = ax.hist(
        weights, 
        bins=bins, 
        color='skyblue', 
        edgecolor='black', 
        linewidth=0.5, 
        alpha=0.7,
        density=False  # 显示频次而非密度
    )
    
    # 标题 & 轴标签
    ax.set_title(f"{layer_name} Weight Distribution", fontsize=12, pad=15)
    ax.set_xlabel("Weight Value", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    
    # 添加统计文本框
    stats_text = (f"Mean: {mean_val:.4f}\n"
                  f"Std: {std_val:.4f}\n"
                  f"Min: {min_val:.4f}\n"
                  f"Max: {max_val:.4f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="w", alpha=0.9))
    
    # Step 4: 叠加高斯拟合曲线（可选）
    if show_gaussian_fit:
        # 拟合高斯分布
        mu, sigma = norm.fit(weights)
        
        # 生成高斯曲线的 x 值
        x = np.linspace(min_val, max_val, 1000)
        y = norm.pdf(x, mu, sigma)
        
        # 将 PDF 缩放到与直方图频次匹配
        bin_width = bins_edges[1] - bins_edges[0]
        y_scaled = y * len(weights) * bin_width  # 使面积 = 总样本数
        
        ax.plot(x, y_scaled, 'r-', linewidth=2, label=f'Gaussian Fit\n$\mu={mu:.4f}, \sigma={sigma:.4f}$')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    # 保存
    if save_fig:
        safe_name = f"weight_dist_{layer_name.replace('.', '_')}"
        plt.savefig(f"{safe_name}.pdf", bbox_inches="tight")
        plt.savefig(f"{safe_name}.png", bbox_inches="tight", dpi=300)
        print(f"✅ Saved weight distribution plot to {safe_name}.pdf/png")


def visualize_subspace_alignment(Delta_Res, W0, layer_name, topk=64):
    # SVD decomposition
    U_d, S_d, V_d = torch.linalg.svd(Delta_Res, full_matrices=False)
    U_w, S_w, V_w = torch.linalg.svd(W0, full_matrices=False)
    
    # 取前 topk 主方向
    U_d_k = U_d[:, :topk]
    U_w_k = U_w[:, :topk]

    # 计算子空间对齐矩阵 (principal angles)
    M = U_w_k.T @ U_d_k
    U_m, S_m, V_m = torch.linalg.svd(M)
    cos_thetas = S_m.clamp(max=1.0)
    thetas = torch.acos(cos_thetas) * 180 / torch.pi  # 转为角度制

    # 可视化 1: 主角分布
    plt.figure(figsize=(6, 4))
    plt.plot(thetas.cpu().numpy(), marker='o')
    plt.title("Principal Angles between subspaces of ΔRes and W₀")
    plt.xlabel("Component index")
    plt.ylabel("Angle (degrees)")
    plt.grid(True)
    plt.show()

    # 可视化 2: 方向相关矩阵 —— 优化布局版本
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))  # 稍微缩小高度

    C = torch.abs(U_w_k.T @ U_d_k)
    im = ax.imshow(C.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)

    ax.set_title("Cosine Similarity Matrix of Subspace Bases")
    ax.set_xlabel("$W-pre$ basis")
    ax.set_ylabel("$W-final$ basis")

    # 创建 colorbar 并调整其位置和大小
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('|cos(angle)|')

    # 调整布局，防止标题/标签被截断
    plt.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.15)

    # 汇总指标
    mean_cos = cos_thetas.mean().item()
    print(f"Average subspace alignment cosθ: {mean_cos:.4f}")

    safe_name = f"projection_{layer_name.replace('.', '_')}.png"
    plt.savefig(safe_name, dpi=300, bbox_inches='tight')  # 确保不裁剪
    print(f"✅ Saved projection plot to {safe_name}")
    plt.close(fig)  # 避免内存泄漏

def visualize_subspace_alignment_and_conditioning(Delta_Res, W0, layer_name, topk=64):
    # ===== Step 1: SVD for both matrices =====
    U_d, S_d, V_d = torch.linalg.svd(Delta_Res, full_matrices=False)
    U_w, S_w, V_w = torch.linalg.svd(W0, full_matrices=False)

    # Move to CPU for plotting
    S_d = S_d.cpu().numpy()
    S_w = S_w.cpu().numpy()

    # ===== Step 2: Analyze conditioning =====
    eps = np.finfo(S_d.dtype).eps
    cond_d = S_d[0] / (S_d[-1] + eps)
    cond_w = S_w[0] / (S_w[-1] + eps)

    energy_d_k = np.sum(S_d[:topk]**2) / np.sum(S_d**2)
    energy_w_k = np.sum(S_w[:topk]**2) / np.sum(S_w**2)

    # ===== Step 3: Plot horizontally: [Singular Values | Subspace Alignment] =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})

    # --- Plot 1: Singular Value Decay (log-log) ---
    colors = sns.color_palette("Set2", 2)  # 或 "Dark2", "husl"

    ax1.loglog(np.arange(1, len(S_w)+1), S_w, marker='o', markersize=3,
               color=colors[0], label=r'$W_{pre}$', alpha=0.85, linewidth=2)
    ax1.loglog(np.arange(1, len(S_d)+1), S_d, marker='s', markersize=3,
               color=colors[1], label=r'$\bar W$', alpha=0.85, linewidth=2)
    ax1.axhline(y=eps, color='gray', linestyle='--', linewidth=1.5, label=r'Machine $\epsilon$')

    ax1.set_xlabel('Singular value index')
    ax1.set_ylabel('Singular value (log scale)')
    ax1.set_title(f'Singular Value Decay: {layer_name}')
    ax1.grid(True, which="both", ls="-.", alpha=0.7)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', fontsize=10)

    # ✅ 替换：带颜色的双行条件数标注（图内左下角）
    # 获取颜色（确保 colors 已定义）
    color_w = colors[0]   # W_pre 的颜色
    color_wbar = colors[1]  # Ŵ 的颜色

    # 第一行：W_pre
    ax1.text(
        0.05, 0.08, f"$\\kappa(W_{{pre}}) = {cond_w:.2e}$",
        transform=ax1.transAxes,
        fontsize=9,
        color=color_w,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color_w, alpha=0.85),
        zorder=10
    )
    # 第二行：Ŵ
    ax1.text(
        0.05, 0.03, f"$\\kappa(\\bar W) = {cond_d:.2e}$",
        transform=ax1.transAxes,
        fontsize=9,
        color=color_wbar,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color_wbar, alpha=0.85),
        zorder=10
    )
    # --- Plot 2: Subspace alignment heatmap ---
    U_d_k = U_d[:, :topk].cpu()
    U_w_k = U_w[:, :topk].cpu()

    C = torch.abs(U_w_k.T @ U_d_k).numpy()
    im = ax2.imshow(C, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title(f"Cosine Similarity of Top-{topk} Bases")
    ax2.set_xlabel(r"$\bar W$ basis")
    ax2.set_ylabel(r"$W_{pre}$ basis")

    # Add colorbar to the right of ax2
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r'$| \cos(\theta) |$', rotation=270, labelpad=10)

    # ✅ 调整布局：增加底部边距，为下方文本留空间
    plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.15, wspace=0.25)

    # Save
    safe_name = f"conditioning_and_projection_{layer_name.replace('.', '_')}.png"
    plt.savefig(safe_name, dpi=400, bbox_inches='tight')
    plt.savefig(f"{safe_name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # ===== Print diagnostics =====
    print(f"📊 Conditioning Report for layer: {layer_name}")
    print(f"  W₀:   σ_max={S_w[0]:.3e}, σ_min={S_w[-1]:.3e}, cond={cond_w:.2e}, Top-{topk} energy={energy_w_k:.1%}")
    print(f"  W_bar: σ_max={S_d[0]:.3e}, σ_min={S_d[-1]:.3e}, cond={cond_d:.2e}, Top-{topk} energy={energy_d_k:.1%}")
    
    if cond_w > 1e8:
        print("  ⚠️  W₀ is severely ill-conditioned!")
    if cond_d > 1e8:
        print("  ⚠️  ΔRes is severely ill-conditioned!")
    if S_w[-1] < 10 * eps:
        print("  ⚠️  W₀ has singular values near machine precision — numerical instability likely.")
    if S_d[-1] < 10 * eps:
        print("  ⚠️  ΔRes has singular values near machine precision — numerical instability likely.")
    
    print(f"✅ Saved combined plot to {safe_name}")

def plot_multi_singular_spectrum(
    tensor_list,
    dataset_names,
    layer_name: str = "Layer",
    k_frac = 0.3,
    figsize = (10, 7),
    log_scale = False,
    max_points_to_plot = 500,
    highlight_names = ["Merge", "Merge + Res"],
):
    """
    对一系列张量进行SVD分解，并可视化它们的奇异值谱。

    Args:
        tensor_list (List[torch.Tensor]): 需要进行SVD分解的张量列表。
        dataset_names (List[str]): 与tensor_list对应的名称，用于图例。
        layer_name (str): 图表的标题，通常是层名称。
        k_frac (float): 一个0到1之间的比例，用于绘制一条垂直线，
                        表示奇异值截断的位置。
        figsize (tuple): 图表的大小。
        log_scale (bool): 是否在Y轴上使用对数尺度。
    """
    if len(tensor_list) != len(dataset_names):
        print(len(tensor_list))
        print(len(dataset_names))
        raise ValueError("tensor_list 和 dataset_names 的长度必须一致。")

    # --- 设置更精细的图表风格 ---
    sns.set_theme(
        style="ticks", # 使用 ticks 风格，比 whitegrid 更简洁
        font_scale=1.2,
        rc={
            "font.family": "serif", "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix", "axes.labelsize": 14,
            "axes.titlesize": 16, "xtick.labelsize": 12,
            "ytick.labelsize": 12, "legend.fontsize": 10,
            "axes.linewidth": 1.2, "grid.linestyle": '--', "grid.alpha": 0.5,
        }
    )

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # --- 准备多样化的绘图样式 ---
    num_curves = len(tensor_list)
    colors = sns.color_palette("husl", n_colors=num_curves)
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    linestyles = ['-', '--', '-.', ':']

    max_rank = 0
    
    # 将高亮和非高亮的曲线分开，先画非高亮的作为背景
    if highlight_names is None:
        highlight_names = []
    
    plot_order = sorted(
        range(num_curves),
        key=lambda i: dataset_names[i] in highlight_names
    )

    # --- 遍历每个张量，计算SVD并绘图 ---
    for i in plot_order:
        tensor, name = tensor_list[i], dataset_names[i]
        
        if tensor.ndim < 2:
            print(f"跳过 '{name}'，因为其维度小于2。")
            continue
            
        tensor_float = tensor
        try:
            S = torch.linalg.svdvals(tensor_float)
        except torch.linalg.LinAlgError as e:
            print(f"对 '{name}' 进行SVD时出错: {e}。跳过。")
            continue

        indices = np.arange(len(S))
        singular_values = S.cpu().numpy()
        
        # --- 智能降采样 ---
        if max_points_to_plot and len(indices) > max_points_to_plot:
            step = len(indices) // max_points_to_plot
            plot_indices = indices[::step]
            plot_values = singular_values[::step]
        else:
            plot_indices = indices
            plot_values = singular_values

        if len(S) > max_rank:
            max_rank = len(S)
            
        # --- 根据是否高亮调整样式 ---
        is_highlighted = name in highlight_names
        
        linewidth = 1.8 if is_highlighted else 1.2
        alpha = 0.9 if is_highlighted else 0.6
        marker_size = 5 if is_highlighted else 4
        zorder = 10 if is_highlighted else 5 # 高亮的在顶层

        # --- 绘制曲线 ---
        ax.plot(
            plot_indices,
            plot_values,
            marker=markers[i % len(markers)] if is_highlighted else 'None', # 非高亮不显示标记点
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i],
            label=name,
            linewidth=linewidth,
            markersize=marker_size,
            alpha=alpha,
            zorder=zorder,
            markevery=max(1, len(plot_indices) // 10) # 每隔一段显示一个标记点
        )

    # --- 绘制 k_frac 对应的垂直线 ---
    if k_frac is not None and max_rank > 0:
        k_index = int(max_rank * k_frac)
        ax.axvline(
            x=k_index, color='gray', linestyle='--', linewidth=1.5,
            label=f'$k$ (frac={k_frac:.2f})', zorder=11 # 确保在最顶层
        )
        # 添加文本标注 k 的位置
        ax.text(k_index, ax.get_ylim()[1]*0.9, f' k={k_index}', color='gray', ha='left')


    # --- 美化图表 ---
    ax.set_title(f'Singular Value Spectrum of {layer_name}')
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Singular Value (Log Scale)')

    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, which="major", axis='y') # 只显示水平网格线，更清爽
    ax.spines['top'].set_visible(False) # 移除上和右的边框
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(left=-max_rank * 0.02, right=max_rank * 1.02)
    if not log_scale:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    safe_name = f"spectrum_{layer_name.replace('.', '_')}_revised"
    plt.savefig(f"{safe_name}.pdf", bbox_inches="tight")
    plt.savefig(f"{safe_name}.png", bbox_inches="tight", dpi=400)
    print(f"✅ Saved revised plot to {safe_name}.pdf/png")

def visualize_model_weight_distribution(
    weight_dict,
    figsize=(8, 5),
    bins=200,
    show_gaussian_fit=True,
    save_fig=True
):
    """
    weight_dict: dict[layer_name -> tensor], 每层权重是 [*, *] 或更高维
    """

    # Step 1: flatten & concat all layers
    all_weights = []

    for layer, W in weight_dict.items():
        all_weights.append(W.reshape(-1).cpu())

    all_weights = torch.cat(all_weights).numpy()

    # Step 2: Statistics
    mean_val = np.mean(all_weights)
    std_val = np.std(all_weights)
    min_val = np.min(all_weights)
    max_val = np.max(all_weights)

    # Step 3: Plot
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    counts, bin_edges, patches = ax.hist(
        all_weights,
        bins=bins,
        color="skyblue",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.75,
        density=False
    )

    ax.set_title("Global Weight Distribution Across All Layers", fontsize=13)
    ax.set_xlabel("Weight Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)

    stat_text = (
        f"Total params: {len(all_weights):,}\n"
        f"Mean: {mean_val:.4f}\n"
        f"Std:  {std_val:.4f}\n"
        f"Min:  {min_val:.4f}\n"
        f"Max:  {max_val:.4f}"
    )
    ax.text(0.02, 0.98, stat_text, transform=ax.transAxes,
            va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="w", alpha=0.9))

    # Step 4: Gaussian fit
    if show_gaussian_fit:
        mu, sigma = norm.fit(all_weights)
        x = np.linspace(min_val, max_val, 1000)
        y = norm.pdf(x, mu, sigma)

        bin_width = bin_edges[1] - bin_edges[0]
        y_scaled = y * len(all_weights) * bin_width

        ax.plot(x, y_scaled, 'r-', lw=2,
                label=f"Gaussian Fit\nμ={mu:.4f}, σ={sigma:.4f}")
        ax.legend(fontsize=9)

    plt.tight_layout()

    # Step 5: Save
    if save_fig:
        plt.savefig("model_global_weight_distribution.pdf", bbox_inches="tight")
        plt.savefig("model_global_weight_distribution.png", bbox_inches="tight", dpi=300)
        print("✅ Saved model-wide distribution plot.")


# ====== 主流程 ======
if __name__ == "__main__":
    arch = infer_model_architecture(model_type)
    print(f"Model type: {model_type} → Architecture: {arch}")
    device = 'cuda'
    # Load task vectors
    task_vectors_path = construct_task_vectors_list(task_vector_dir, other_model_id_list, base_model_id)
    task_vector_list = [TaskVector.load(p, device) for p in task_vectors_path.values()]
    
    T = len(task_vector_list)
    layer_names = []
    gamma_values = []
    
    # ==================== 在主流程中应用过滤 ====================
    # 1. 根据架构获取排除规则
    exclude_patterns = get_exclude_patterns(arch)
    print(f"Applying filter rules for '{arch}' architecture: {exclude_patterns}")
    
    all_layer_names = list(task_vector_list[0].task_vector_param_dict.keys())
    print(f"Found {len(all_layer_names)} total layers. Processing and filtering...")
    
    visualize_model_weight_distribution(task_vector_list[0].task_vector_param_dict)


    for layer_name in all_layer_names:
        # 2. 首先进行基于正则表达式的过滤
        if should_exclude_by_regex(layer_name, exclude_patterns):
            continue
        task_vectors = [tv.task_vector_param_dict[layer_name] for tv in task_vector_list]
        
        # 3. 然后进行基于维度的过滤（只保留 2D 张量）
        if any(vec.ndim != 2 for vec in task_vectors):
            continue
        
        T = len(task_vectors)
        out_dim, in_dim = task_vectors[0].shape

        traces = torch.tensor([torch.trace(W_t.T @ W_t).item() for W_t in task_vectors])
        log_traces = torch.log(traces + 1e-12)
        gamma = torch.var(log_traces) / (torch.mean(log_traces).pow(2) + 1e-12)
        flag = gamma > 0.3
        eps_abs = 2e-4
        avg_trace = traces.mean().item()
        Sigmas = []
        WSigma_sum = torch.zeros_like(task_vectors[0], device=device)
        Sigma_sum  = torch.zeros((in_dim, in_dim), device=device)

        # for W_t in task_vectors:
        #     W_t = W_t - W_t.mean(dim=0, keepdim=True)
        #     Sigma_raw = W_t.T @ W_t

        #     tr = torch.trace(Sigma_raw) + 1e-12

        #     if flag:
        #         Sigma_t = Sigma_raw / tr
        #         eps_t = eps_abs / tr
        #     else:
        #         Sigma_t = Sigma_raw
        #         eps_t = eps_abs

        #     Sigmas.append(Sigma_t)
        #     Sigma_t = Sigma_t + eps_t * torch.eye(in_dim, device=device)

        #     WSigma_sum += W_t @ Sigma_t
        #     Sigma_sum  += Sigma_t

        # C_agg = torch.mean(sum(Sigmas), dim=0, keepdim=True)

        # if flag:
        #     C_agg = C_agg / (avg_trace + 1e-12)

        # A = Sigma_sum + C_agg
        # B = WSigma_sum
        # try:
        #     A_inv = torch.linalg.inv(A)
        # except RuntimeError:
        #     A_inv = torch.linalg.pinv(A)
        # W_0 = B @ A_inv
        # merging_vector = W_0

        # ##################
        # for W_t in task_vectors:
        #     visualize_weight_distribution(W_t, layer_name)

        # ##################

        # if flag:
        #     Sigma_mean = Sigma_sum / T

        #     Sigmas_res = [S_t - Sigma_mean for S_t in Sigmas]

        #     Delta_Res = torch.zeros_like(task_vectors[0], device=device)
        #     for W_t, S_res_t in zip(task_vectors, Sigmas_res):
        #         Delta_Res += W_t @ S_res_t

        #     Delta_Fused = Delta_Res + W_0

        #     # visualize_subspace_alignment(Delta_Fused, W_0, layer_name)
        #     # # 5. 对融合后的向量进行 SVD
        #     U_fused, S_fused, Vh_fused = torch.linalg.svd(Delta_Fused, full_matrices=False)
        #     r = S_fused.shape[0]

        #     k_frac = 0.3
        #     k = int(r * k_frac)

        #     S_fused_k = S_fused[:k]
        #     sigma_iso_fused = S_fused_k.mean()

        #     U_fused_k = U_fused[:, :k]
        #     V_fused_k = Vh_fused[:k, :].T 

        #     merging_vector_new = merging_vector + sigma_iso_fused * (U_fused_k @ V_fused_k.T)
        #     # visualize_subspace_alignment(merging_vector_new, W_0, layer_name)
        #     # visualize_subspace_alignment_and_conditioning(merging_vector_new, W_0, layer_name, topk=64)


            # tensor_list = task_vectors + [merging_vector, merging_vector_new]
            # all_datasets = ['MNIST', 'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'RESISC45', 'SUN397', 'SVHN', 'PCAM', 'CIFAR100', 'STL10', 'OxfordIIITPet', 'Flowers102', 'FER2013', 'CIFAR10', 'Food101', 'RenderedSST2', 'EMNIST', 'FashionMNIST', 'KMNIST']
            # # plot_multi_singular_spectrum(tensor_list, ['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE', "Merge", "Merge + Res"], layer_name)
            # plot_multi_singular_spectrum(tensor_list, all_datasets[:task_nums] + ["Merge", "Merge + Res"], layer_name)

    #     # 只有通过所有过滤的层才会被添加
    #     layer_names.append(layer_name)
    #     device = task_vectors[0].device
    #     traces = torch.tensor([torch.trace(W_t.T @ W_t).item() for W_t in task_vectors], device=device)
    #     log_traces = torch.log(traces + 1e-12)
    #     gamma = torch.var(log_traces) / (torch.mean(log_traces).pow(2) + 1e-12)
    #     gamma_values.append(gamma.item())
    # # ========================================================
    # print(f"Processed and kept {len(layer_names)} layers after filtering.")
    
    # # 检查是否还有剩余的层用于可视化
    # if not layer_names:
    #     print("❌ Error: No layers left after filtering. Cannot generate visualization.")
    #     print("Please check your filter rules and the model's parameter names.")
    # else:
    #     # Build layer type map
    #     layer_type_map = build_layer_type_map(layer_names, arch)
    #     visualize_structural_heterogeneity_cvpr(
    #         layer_names, gamma_values, layer_type_map, 
    #         model_type=model_type, threshold=0.3
    #     )
