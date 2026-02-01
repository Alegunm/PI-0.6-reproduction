"""
绘制两种 VF 架构图：输入与模型分区，无重叠（明确 GAP 与坐标）。
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

R = 0.06
GAP = 0.28   # 区块之间留空，避免重叠


def draw_prev_architecture(out_path="architecture_prev_4b_vla.png"):
    """之前：Input → Model (4B VLM + Value Head) → Output，无重叠."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.2, 9)
    ax.set_ylim(0.2, 4.5)
    ax.set_title("Previous: 4B VLA-based Value Function", fontsize=14, pad=16)

    y0 = 2.9
    h = 0.88

    # ----- 1. Input 区（左，O_t / L_t 框加大） -----
    x_in0, w_in = 0.2, 2.35
    pad_in, gap_o_l = 0.28, 0.22
    wo, ho = 0.88, 0.72
    ax.text(x_in0 + w_in/2, 4.15, "Input", fontsize=12, ha="center", va="bottom", weight="bold")
    input_bg = FancyBboxPatch((x_in0, 2.42), w_in, 1.5, boxstyle=f"round,pad=0.02,rounding_size={R}",
                              facecolor="#f0f0f0", edgecolor="gray", linewidth=1.5, alpha=0.95)
    ax.add_patch(input_bg)
    b_img = FancyBboxPatch((x_in0+pad_in, 2.7), wo, ho, boxstyle=f"round,pad=0.02,rounding_size={R*0.8}",
                           facecolor="#e0e8f0", edgecolor="black", linewidth=1)
    ax.add_patch(b_img)
    ax.text(x_in0+pad_in+wo/2, 2.7+ho/2, "Observation\nO_t (Image)", fontsize=10, ha="center", va="center", multialignment="center")
    b_txt = FancyBboxPatch((x_in0+pad_in+wo+gap_o_l, 2.7), wo, ho, boxstyle=f"round,pad=0.02,rounding_size={R*0.8}",
                           facecolor="#e8f0e0", edgecolor="black", linewidth=1)
    ax.add_patch(b_txt)
    ax.text(x_in0+pad_in+wo+gap_o_l+wo/2, 2.7+ho/2, "Language\nL_t (Prompt)", fontsize=10, ha="center", va="center", multialignment="center")

    # ----- 2. Model 区（中，与 Input 间隔 GAP） -----
    x_m0 = x_in0 + w_in + GAP   # Model 左缘
    w_m_bg = 4.0                # Model 背景宽
    x_m1 = x_m0 + w_m_bg        # Model 右缘

    ax.text(x_m0 + w_m_bg/2, 4.15, "Model", fontsize=12, ha="center", va="bottom", weight="bold")
    model_bg = FancyBboxPatch((x_m0, 2.0), w_m_bg, 1.85, boxstyle=f"round,pad=0.02,rounding_size={R}",
                              facecolor="#e8f4fc", edgecolor="steelblue", linewidth=2, alpha=0.9)
    ax.add_patch(model_bg)
    ax.text(x_m0 + 0.15, 3.72, "#Parameters: ~4B", fontsize=9, ha="left", va="top", style="italic", color="dimgray")

    w_vlm, w_vh = 1.75, 1.25
    x_vlm = x_m0 + 0.2
    x_vh = x_vlm + w_vlm + 0.2   # 与 4B VLM 间隔 0.2

    b1 = FancyBboxPatch((x_vlm, y0 - h/2), w_vlm, h, boxstyle=f"round,pad=0.02,rounding_size={R}",
                        facecolor="#b8d4e8", edgecolor="black", linewidth=1.2)
    ax.add_patch(b1)
    ax.text(x_vlm + w_vlm/2, y0, "4B VLM\n(Vision+Language)\nFrozen / LoRA", fontsize=9, ha="center", va="center", multialignment="center")
    b2 = FancyBboxPatch((x_vh, y0 - h/2), w_vh, h, boxstyle=f"round,pad=0.02,rounding_size={R}",
                        facecolor="#a8e6a0", edgecolor="black", linewidth=1.2)
    ax.add_patch(b2)
    ax.text(x_vh + w_vh/2, y0, "Value Head\n640→201", fontsize=9, ha="center", va="center", multialignment="center")

    # ----- 3. Output 区（右，与 Model 间隔 GAP，完全在 Model 框外） -----
    x_out0 = x_m1 + GAP
    w_out = 1.15

    ax.text(x_out0 + w_out/2, 4.15, "Output", fontsize=12, ha="center", va="bottom", weight="bold")
    b3 = FancyBboxPatch((x_out0, y0 - h/2), w_out, h, boxstyle=f"round,pad=0.02,rounding_size={R}",
                        facecolor="#ffe4b5", edgecolor="black", linewidth=1.2)
    ax.add_patch(b3)
    ax.text(x_out0 + w_out/2, y0, "Value\n(201 bins)", fontsize=9, ha="center", va="center", multialignment="center")

    # 箭头：右缘→左缘，不重叠
    ax.annotate("", xy=(x_m0, y0), xytext=(x_in0 + w_in, y0), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(x_vh, y0), xytext=(x_vlm + w_vlm, y0), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(x_out0, y0), xytext=(x_vh + w_vh, y0), arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def draw_current_architecture(out_path="architecture_current_stitched.png"):
    """当前：Input → Model (SigLIP→Projector→Gemma→Value Head) → Output，无重叠."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.2, 12)
    ax.set_ylim(0.2, 4.5)
    ax.set_title("Current: Stitched VF (SigLIP + Projector + Gemma 270M + Value Head)", fontsize=13, pad=16)

    y0 = 2.9
    h = 0.88
    gap_in = 0.22   # 模型内部块间距

    # ----- 1. Input 区（O_t / L_t 框加大） -----
    x_in0, w_in = 0.2, 2.35
    pad_in, gap_o_l = 0.28, 0.22
    wo, ho = 0.88, 0.72
    ax.text(x_in0 + w_in/2, 4.15, "Input", fontsize=12, ha="center", va="bottom", weight="bold")
    input_bg = FancyBboxPatch((x_in0, 2.42), w_in, 1.5, boxstyle=f"round,pad=0.02,rounding_size={R}",
                              facecolor="#f0f0f0", edgecolor="gray", linewidth=1.5, alpha=0.95)
    ax.add_patch(input_bg)
    b_img = FancyBboxPatch((x_in0+pad_in, 2.7), wo, ho, boxstyle=f"round,pad=0.02,rounding_size={R*0.8}",
                           facecolor="#e0e8f0", edgecolor="black", linewidth=1)
    ax.add_patch(b_img)
    ax.text(x_in0+pad_in+wo/2, 2.7+ho/2, "Observation\nO_t (Image)", fontsize=10, ha="center", va="center", multialignment="center")
    b_txt = FancyBboxPatch((x_in0+pad_in+wo+gap_o_l, 2.7), wo, ho, boxstyle=f"round,pad=0.02,rounding_size={R*0.8}",
                           facecolor="#e8f0e0", edgecolor="black", linewidth=1)
    ax.add_patch(b_txt)
    ax.text(x_in0+pad_in+wo+gap_o_l+wo/2, 2.7+ho/2, "Language\nL_t (Prompt)", fontsize=10, ha="center", va="center", multialignment="center")

    # ----- 2. Model 区（左缘与 Input 间隔 GAP） -----
    x_m0 = x_in0 + w_in + GAP
    # 内部块：SigLIP, Projector, Gemma, Value Head，每块间 gap_in
    w1, w2, w3, w4 = 1.05, 0.9, 1.35, 1.0
    w_m_bg = 0.25 + w1 + gap_in + w2 + gap_in + w3 + gap_in + w4 + 0.25  # 内边距 0.25
    x_m1 = x_m0 + w_m_bg

    ax.text(x_m0 + w_m_bg/2, 4.15, "Model", fontsize=12, ha="center", va="bottom", weight="bold")
    model_bg = FancyBboxPatch((x_m0, 2.0), w_m_bg, 1.85, boxstyle=f"round,pad=0.02,rounding_size={R}",
                              facecolor="#e8f4fc", edgecolor="steelblue", linewidth=2, alpha=0.9)
    ax.add_patch(model_bg)
    ax.text(x_m0 + 0.2, 3.72, "#Parameters: ~670M", fontsize=8, ha="left", va="top", style="italic", color="dimgray")

    x1 = x_m0 + 0.25
    x2 = x1 + w1 + gap_in
    x3 = x2 + w2 + gap_in
    x4 = x3 + w3 + gap_in

    b1 = FancyBboxPatch((x1, y0 - h/2), w1, h, boxstyle=f"round,pad=0.02,rounding_size={R}",
                        facecolor="#e6e6fa", edgecolor="black", linewidth=1.2)
    ax.add_patch(b1)
    ax.text(x1 + w1/2, y0, "SigLIP\nVision\n1152, Frozen", fontsize=8, ha="center", va="center", multialignment="center")
    b2 = FancyBboxPatch((x2, y0 - h/2), w2, h, boxstyle=f"round,pad=0.02,rounding_size={R}",
                        facecolor="#fffacd", edgecolor="black", linewidth=1.2)
    ax.add_patch(b2)
    ax.text(x2 + w2/2, y0, "Projector\n1152→640", fontsize=8, ha="center", va="center", multialignment="center")
    b3 = FancyBboxPatch((x3, y0 - h/2), w3, h, boxstyle=f"round,pad=0.02,rounding_size={R}",
                        facecolor="#b8d4e8", edgecolor="black", linewidth=1.2)
    ax.add_patch(b3)
    ax.text(x3 + w3/2, y0, "Gemma 3\n270M-it\nLoRA/Frozen", fontsize=8, ha="center", va="center", multialignment="center")
    b4 = FancyBboxPatch((x4, y0 - h/2), w4, h, boxstyle=f"round,pad=0.02,rounding_size={R}",
                        facecolor="#a8e6a0", edgecolor="black", linewidth=1.2)
    ax.add_patch(b4)
    ax.text(x4 + w4/2, y0, "Value Head\n640→201", fontsize=8, ha="center", va="center", multialignment="center")

    # ----- 3. Output 区（在 Model 框右侧外，间隔 GAP） -----
    x_out0 = x_m1 + GAP
    w_out = 1.05

    ax.text(x_out0 + w_out/2, 4.15, "Output", fontsize=12, ha="center", va="bottom", weight="bold")
    b5 = FancyBboxPatch((x_out0, y0 - h/2), w_out, h, boxstyle=f"round,pad=0.02,rounding_size={R}",
                        facecolor="#ffe4b5", edgecolor="black", linewidth=1.2)
    ax.add_patch(b5)
    ax.text(x_out0 + w_out/2, y0, "Value\n(201 bins)", fontsize=8, ha="center", va="center", multialignment="center")

    # 箭头
    ax.annotate("", xy=(x_m0, y0), xytext=(x_in0 + w_in, y0), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(x2, y0), xytext=(x1 + w1, y0), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(x3, y0), xytext=(x2 + w2, y0), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(x4, y0), xytext=(x3 + w3, y0), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(x_out0, y0), xytext=(x4 + w4, y0), arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    draw_prev_architecture()
    draw_current_architecture()
