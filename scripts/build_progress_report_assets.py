#!/usr/bin/env python3
"""Build CPU-only, PPT-ready figures for the precipitation inversion report.

The script deliberately reads only already-completed metadata/evaluation files.
It neither loads a neural-network checkpoint nor uses CUDA, so it can run while
another experiment occupies the GPUs.  Every generated slide uses a 16:9 canvas
and records its source files in ``figure_sources.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "progress_report_20260901"

NAVY = "#16324F"
BLUE = "#2F6BFF"
CYAN = "#24A8C7"
GREEN = "#2A9D6F"
ORANGE = "#F4A261"
RED = "#D9534F"
PURPLE = "#7457C8"
GOLD = "#D9A520"
INK = "#202A35"
MUTED = "#64748B"
LIGHT = "#F3F6FA"
GRID = "#D8E0E9"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(relative: str | Path) -> dict[str, Any]:
    path = PROJECT_ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "WenQuanYi Zen Hei",
                "Droid Sans Fallback",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )


def slide(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.045, 0.945, title, fontsize=23, weight="bold", color=NAVY, va="top")
    if subtitle:
        ax.text(0.047, 0.895, subtitle, fontsize=10.5, color=MUTED, va="top")
    ax.plot([0.045, 0.955], [0.865, 0.865], color=GRID, lw=1.2)
    return fig, ax


def save(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    *,
    face: str = LIGHT,
    edge: str = GRID,
    title_color: str = NAVY,
    body_color: str = INK,
    title_size: float = 13,
    body_size: float = 9.5,
    linewidth: float = 1.3,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * (0.64 if body else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        weight="bold",
        color=title_color,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.29,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=body_color,
            linespacing=1.35,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    width: float = 1.8,
    style: str = "-|>",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=14,
            linewidth=width,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def metric_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    note: str,
    color: str,
) -> None:
    rounded_box(ax, x, y, w, h, "", face="#FAFCFE", edge=GRID)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.012,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.text(x + 0.025, y + h * 0.72, label, color=MUTED, fontsize=9.5, va="center")
    ax.text(x + 0.025, y + h * 0.42, value, color=NAVY, fontsize=20, weight="bold", va="center")
    ax.text(x + 0.025, y + h * 0.15, note, color=MUTED, fontsize=8.5, va="center")


def draw_overall_route(out: Path, dpi: int) -> None:
    fig, ax = slide(
        "项目总体路线：把星载DPR知识迁移到地基GR输入",
        "最终目标不是复现一张二维降水图，而是在统一三维网格中恢复每个高度层的降水率。",
    )
    rounded_box(
        ax,
        0.055,
        0.58,
        0.17,
        0.17,
        "地基雷达 GR",
        "稀疏三维反射率\n部署时真正可获得",
        face="#EAF3FF",
        edge=BLUE,
    )
    rounded_box(
        ax,
        0.30,
        0.55,
        0.20,
        0.23,
        "Stage 2",
        "GR稀疏反射率\n→ DPR反射率与支持域\n当前主要瓶颈",
        face="#FFF4E8",
        edge=ORANGE,
        title_color="#A64B00",
    )
    rounded_box(
        ax,
        0.575,
        0.58,
        0.15,
        0.17,
        "DPR反射率",
        "统一到卫星数值域\n及其有效回波范围",
        face="#F2EDFF",
        edge=PURPLE,
    )
    rounded_box(
        ax,
        0.79,
        0.55,
        0.15,
        0.23,
        "Stage 1",
        "DPR反射率\n→ 三维降水率\n已基本封版",
        face="#EAF8F1",
        edge=GREEN,
        title_color="#146C43",
    )
    arrow(ax, (0.225, 0.665), (0.30, 0.665), color=BLUE)
    arrow(ax, (0.50, 0.665), (0.575, 0.665), color=ORANGE)
    arrow(ax, (0.725, 0.665), (0.79, 0.665), color=PURPLE)

    rounded_box(
        ax,
        0.30,
        0.245,
        0.20,
        0.17,
        "卫星 DPR反射率标签",
        "训练Stage 2",
        face="#F8F4FF",
        edge=PURPLE,
    )
    rounded_box(
        ax,
        0.575,
        0.245,
        0.20,
        0.17,
        "卫星 pre_dpr降水标签",
        "训练Stage 1 / 评价最终结果",
        face="#EDF9F3",
        edge=GREEN,
    )
    arrow(ax, (0.40, 0.415), (0.40, 0.55), color=PURPLE)
    arrow(ax, (0.675, 0.415), (0.865, 0.55), color=GREEN, connectionstyle="arc3,rad=-0.18")
    ax.text(
        0.055,
        0.115,
        "当前结论",
        color=NAVY,
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.145,
        0.115,
        "Stage 1单独输入真实DPR反射率时表现较好；一旦换成Stage 2预测反射率，误差叠加使最终降水明显劣化。",
        color=RED,
        fontsize=12,
        weight="bold",
    )
    save(fig, out / "figures/01_overall_task_route.png", dpi=dpi)


def draw_split_and_patch(split: Mapping[str, Any], indices: Mapping[str, Mapping[str, Any]], out: Path, dpi: int) -> None:
    fig, ax = slide(
        "数据如何进入模型：先按日期分轨道，再切上下文Patch",
        "轨道级划分避免同一天的相邻观测同时出现在训练集和验证/测试集，降低信息泄漏。",
    )
    labels = ["训练集", "验证集", "测试集"]
    keys = ["train", "val", "test"]
    colors_ = [BLUE, ORANGE, GREEN]
    file_counts = [int(split["splits"][k]["totals"]["file_count"]) for k in keys]
    group_counts = [int(split["splits"][k]["group_count"]) for k in keys]
    patch_counts = [int(indices[k]["patch_count"]) for k in keys]

    bar_ax = fig.add_axes([0.055, 0.54, 0.34, 0.27])
    x = np.arange(3)
    bars = bar_ax.bar(x, file_counts, color=colors_, width=0.58)
    bar_ax.set_xticks(x, labels)
    bar_ax.set_ylabel("轨道文件数")
    bar_ax.set_title("254条轨道的70% / 15% / 15%划分", loc="left", color=NAVY, weight="bold", pad=10)
    bar_ax.set_ylim(0, 225)
    bar_ax.spines[["top", "right"]].set_visible(False)
    bar_ax.grid(axis="y", color=GRID, alpha=0.6)
    for b, f, g, p in zip(bars, file_counts, group_counts, patch_counts):
        bar_ax.text(b.get_x() + b.get_width() / 2, f + 4, f"{f}轨\n{g}天组\n{p}个Patch", ha="center", fontsize=9)

    rounded_box(
        ax,
        0.46,
        0.61,
        0.17,
        0.16,
        "原始整轨",
        "(nscan, 49, 60)\nnscan=85～638，不定长",
        face="#F7F9FC",
    )
    rounded_box(
        ax,
        0.685,
        0.61,
        0.25,
        0.16,
        "滑窗切分",
        "32个scan为不重叠输出核心\n前后各16个scan作为上下文",
        face="#EAF3FF",
        edge=BLUE,
    )
    arrow(ax, (0.63, 0.69), (0.685, 0.69), color=BLUE)

    # Patch strip: halo / core / halo.
    strip_y = 0.39
    widths = [0.12, 0.24, 0.12]
    starts = [0.46, 0.58, 0.82]
    strip_colors = ["#DCEAFF", BLUE, "#DCEAFF"]
    texts = ["左上下文\n16 scan", "唯一输出核心\n32 scan", "右上下文\n16 scan"]
    for sx, sw, fc, text_ in zip(starts, widths, strip_colors, texts):
        ax.add_patch(plt.Rectangle((sx, strip_y), sw, 0.10, facecolor=fc, edgecolor=WHITE, lw=2))
        ax.text(sx + sw / 2, strip_y + 0.05, text_, ha="center", va="center", color=WHITE if fc == BLUE else NAVY, fontsize=9, weight="bold")
    ax.text(0.46, 0.515, "模型实际看到64个scan，但只保留中央32个scan，减轻块边界伪影", color=NAVY, fontsize=10.5, weight="bold")

    rounded_box(
        ax,
        0.055,
        0.17,
        0.24,
        0.18,
        "水平方向补齐",
        "(64,49,60) → (64,64,60)\n只补ray方向15列；高度不补",
        face="#FFF8EC",
        edge=ORANGE,
    )
    rounded_box(
        ax,
        0.365,
        0.17,
        0.24,
        0.18,
        "单个样本张量",
        "Stage 1: (3,64,64,60)\nStage 2: (4,64,64,60)",
        face="#F3F0FF",
        edge=PURPLE,
    )
    rounded_box(
        ax,
        0.675,
        0.17,
        0.26,
        0.18,
        "组成训练Batch",
        "(B,C,64,64,60)\n当前每卡B=1，多卡同步梯度",
        face="#EDF9F3",
        edge=GREEN,
    )
    arrow(ax, (0.295, 0.26), (0.365, 0.26), color=ORANGE)
    arrow(ax, (0.605, 0.26), (0.675, 0.26), color=PURPLE)
    save(fig, out / "figures/02_dataset_split_and_patch.png", dpi=dpi)


def draw_unet_levels(ax: plt.Axes, *, input_channels: int, stage: int) -> None:
    y_values = [0.68, 0.56, 0.44, 0.32, 0.20]
    x_enc = [0.19, 0.29, 0.39, 0.49, 0.59]
    x_dec = [0.69, 0.77, 0.85, 0.93]
    channels = [16, 32, 64, 128, 256]
    spatial = [64, 32, 16, 8, 4]
    level_colors = ["#DDEBFF", "#C7DCFF", "#A8C9FF", "#84AFFF", "#5D8DF0"]

    rounded_box(
        ax,
        0.03,
        0.60,
        0.105,
        0.15,
        f"输入 {input_channels}通道",
        f"(B,{input_channels},64,64,60)",
        face="#F7F9FC",
        title_size=11,
        body_size=8.5,
    )
    arrow(ax, (0.135, 0.675), (0.18, 0.675), color=BLUE)

    for i, (x, y, c, s, fc) in enumerate(zip(x_enc, y_values, channels, spatial, level_colors)):
        rounded_box(
            ax,
            x - 0.04,
            y - 0.052,
            0.08,
            0.104,
            f"C={c}",
            f"{s}×{s}×60",
            face=fc,
            edge=BLUE,
            title_size=10.5,
            body_size=8.5,
        )
        if i:
            arrow(ax, (x_enc[i - 1] + 0.04, y_values[i - 1] - 0.01), (x - 0.04, y + 0.01), color=BLUE)

    dec_channels = [128, 64, 32, 16]
    dec_spatial = [8, 16, 32, 64]
    for i, (x, c, s) in enumerate(zip(x_dec, dec_channels, dec_spatial)):
        y = y_values[3 - i]
        rounded_box(
            ax,
            x - 0.035,
            y - 0.052,
            0.07,
            0.104,
            f"C={c}",
            f"{s}×{s}×60",
            face="#E8F7F0",
            edge=GREEN,
            title_size=10,
            body_size=8.2,
        )
        previous = (x_enc[-1] + 0.04, y_values[-1] + 0.01) if i == 0 else (x_dec[i - 1] + 0.035, y_values[4 - i] + 0.01)
        arrow(ax, previous, (x - 0.035, y - 0.005), color=GREEN)
        # Skip connection from matching encoder resolution.
        enc_index = 3 - i
        arrow(
            ax,
            (x_enc[enc_index], y_values[enc_index] + 0.055),
            (x, y + 0.055),
            color=PURPLE,
            width=1.1,
            connectionstyle="arc3,rad=-0.18",
        )

    ax.text(0.39, 0.795, "编码器：只在scan/ray方向下采样 (2,2,1)", color=BLUE, fontsize=11, weight="bold", ha="center")
    ax.text(0.79, 0.795, "解码器：水平上采样＋同尺度跳跃连接", color=GREEN, fontsize=11, weight="bold", ha="center")
    ax.text(
        0.40,
        0.12,
        "各层高度恒为60：不做高度池化或padding",
        color=RED,
        fontsize=10.5,
        weight="bold",
        ha="center",
    )


def draw_stage1_architecture(out: Path, dpi: int) -> None:
    fig, ax = slide(
        "Stage 1最终架构：保持高度的3D U-Net＋降水类型辅助头",
        "输入是真实DPR反射率；主任务预测log1p降水，辅助任务从三维形态判断层云/对流/其他。",
    )
    draw_unet_levels(ax, input_channels=3, stage=1)
    rounded_box(
        ax,
        0.80,
        0.265,
        0.17,
        0.105,
        "降水输出头",
        "1×1×1卷积\n(B,1,64,64,60)",
        face="#EDF9F3",
        edge=GREEN,
        title_size=10.0,
        body_size=8.0,
    )
    arrow(ax, (0.93, 0.625), (0.885, 0.37), color=GREEN, connectionstyle="arc3,rad=-0.12")
    rounded_box(
        ax,
        0.69,
        0.055,
        0.27,
        0.135,
        "三维有序形态分类头 T3D",
        "保留高度顺序的3D卷积 → 压缩高度 → 2D分类\n输出(B,3,64,64)，typePrecip只作标签，不作输入",
        face="#F3F0FF",
        edge=PURPLE,
        title_size=10.5,
        body_size=8.2,
    )
    arrow(ax, (0.93, 0.50), (0.82, 0.19), color=PURPLE, connectionstyle="arc3,rad=0.25")
    ax.text(0.055, 0.085, "约202万参数；主干输出通道16/32/64/128/256", color=MUTED, fontsize=9.5)
    save(fig, out / "figures/03_stage1_architecture.png", dpi=dpi)


def draw_stage2_architecture(out: Path, dpi: int) -> None:
    fig, ax = slide(
        "Stage 2最终第一版架构：共享3D U-Net主干＋两个独立输出头",
        "同一组空间特征同时回答两个问题：这里有没有DPR回波？如果有，反射率是多少？",
    )
    draw_unet_levels(ax, input_channels=4, stage=2)
    rounded_box(
        ax,
        0.80,
        0.265,
        0.17,
        0.105,
        "Support头",
        "DPR回波概率\n(B,1,64,64,60)",
        face="#FFF8EC",
        edge=ORANGE,
        title_size=10.0,
        body_size=8.0,
    )
    arrow(ax, (0.93, 0.625), (0.885, 0.37), color=ORANGE, connectionstyle="arc3,rad=-0.12")
    rounded_box(
        ax,
        0.80,
        0.055,
        0.17,
        0.125,
        "反射率头 dBZ",
        "标准化DPR dBZ\n(B,1,64,64,60)",
        face="#F3F0FF",
        edge=PURPLE,
        title_size=10.5,
        body_size=8.3,
    )
    arrow(ax, (0.93, 0.50), (0.885, 0.18), color=PURPLE, connectionstyle="arc3,rad=0.25")
    ax.text(
        0.055,
        0.055,
        "输入通道：①标准化GR稀疏dBZ  ②GR物理有效mask  ③最近GR观测距离[0,1]  ④高度[-1,1]；约199万参数",
        color=MUTED,
        fontsize=9.5,
    )
    save(fig, out / "figures/04_stage2_architecture.png", dpi=dpi)


def draw_masks(out: Path, dpi: int) -> None:
    fig, ax = slide(
        "Mask不是“把数据变成0”：它决定输入含义、监督范围和评价口径",
        "缺测值填0只为了组成稠密张量；独立mask负责告诉模型/损失这个0是否有物理意义。",
    )
    rows = [
        ("geometry/core", "真实轨道位置且位于唯一输出核心", "排除轨道边界halo与水平padding", BLUE),
        ("DPR valid", "DPR反射率是有限、非填充值", "Stage 1输入有效通道；不是零降水", PURPLE),
        ("Stage 1 loss", "core ∩ DPR valid ∩ 非CFB杂波 ∩ 正降水QC", "只在可靠的反射率—降水配对处回归", GREEN),
        ("type loss", "core内typePrecip属于层云/对流/其他", "只监督二维廓线类别，不输入标签", CYAN),
        ("GR value", "GR稀疏dBZ是有限、非-9999填充值", "Stage 2输入mask；普通负dBZ仍是有效值", ORANGE),
        ("M_support", "core ∩ pre_dpr为有限且≥0", "支持域BCE的可信标签域；含真零背景", RED),
        ("y_support", "DPR dBZ物理有效", "在M_support内：1=有DPR回波，0=无回波", RED),
        ("M_dbz", "core ∩ DPR dBZ物理有效", "dBZ回归范围；必然是y_support=1", PURPLE),
        ("Q11/gap/outside", "按GR直接值/插值代理对DPR目标分区", "只用于训练统计和诊断，不能作为部署输入", NAVY),
    ]
    y0 = 0.79
    row_h = 0.071
    headers = ["名称", "什么时候为1", "在哪里使用"]
    xs = [0.055, 0.23, 0.60]
    widths = [0.16, 0.35, 0.34]
    for x, w, h in zip(xs, widths, headers):
        ax.add_patch(plt.Rectangle((x, y0), w, 0.055, facecolor=NAVY, edgecolor=WHITE))
        ax.text(x + w / 2, y0 + 0.0275, h, ha="center", va="center", color=WHITE, weight="bold", fontsize=10)
    for i, (name, condition, use, color_) in enumerate(rows):
        y = y0 - (i + 1) * row_h
        fc = WHITE if i % 2 == 0 else "#F7F9FC"
        for x, w in zip(xs, widths):
            ax.add_patch(plt.Rectangle((x, y), w, row_h, facecolor=fc, edgecolor=GRID, lw=0.7))
        ax.text(xs[0] + 0.012, y + row_h / 2, name, va="center", color=color_, weight="bold", fontsize=9.5)
        ax.text(xs[1] + 0.012, y + row_h / 2, condition, va="center", color=INK, fontsize=9.1)
        ax.text(xs[2] + 0.012, y + row_h / 2, use, va="center", color=INK, fontsize=9.1)
    ax.text(0.055, 0.055, "关键原则：被mask排除的格点不产生loss，也不产生梯度；它们不是自动被当成‘真实0’。", color=RED, fontsize=11.5, weight="bold")
    save(fig, out / "figures/05_masks_and_supervision.png", dpi=dpi)


def draw_loss_flow(out: Path, dpi: int) -> None:
    fig, ax = slide(
        "一个总损失如何训练双头网络：先分别算错，再相加后统一反向传播",
        "两个输出头不是只靠同一个误差硬凑；每个头都有自己的目标、mask和损失。",
    )
    rounded_box(ax, 0.055, 0.56, 0.16, 0.18, "共享U-Net特征 F", "(B,16,64,64,60)", face="#EAF3FF", edge=BLUE)
    rounded_box(ax, 0.29, 0.67, 0.18, 0.14, "Support头", "预测DPR回波概率", face="#FFF4E8", edge=ORANGE)
    rounded_box(ax, 0.29, 0.43, 0.18, 0.14, "dBZ头", "预测标准化DPR反射率", face="#F3F0FF", edge=PURPLE)
    arrow(ax, (0.215, 0.65), (0.29, 0.74), color=ORANGE)
    arrow(ax, (0.215, 0.65), (0.29, 0.50), color=PURPLE)
    rounded_box(ax, 0.55, 0.67, 0.19, 0.14, "Lsupport", "带pos_weight=5的BCE\n只在M_support内", face="#FFF9F2", edge=ORANGE)
    rounded_box(ax, 0.55, 0.43, 0.19, 0.14, "Ldbz", "强回波加权Smooth-L1\n只在M_dbz内", face="#F8F5FF", edge=PURPLE)
    arrow(ax, (0.47, 0.74), (0.55, 0.74), color=ORANGE)
    arrow(ax, (0.47, 0.50), (0.55, 0.50), color=PURPLE)
    rounded_box(ax, 0.80, 0.54, 0.15, 0.19, "总损失", "L = 1·Lsupport\n    + 1·Ldbz", face="#EDF9F3", edge=GREEN, title_color="#146C43")
    arrow(ax, (0.74, 0.74), (0.80, 0.65), color=ORANGE)
    arrow(ax, (0.74, 0.50), (0.80, 0.61), color=PURPLE)
    arrow(ax, (0.875, 0.54), (0.135, 0.54), color=GREEN, width=2.5, connectionstyle="arc3,rad=-0.32")
    ax.text(0.50, 0.285, "反向传播时梯度自动满足", ha="center", color=NAVY, fontsize=12, weight="bold")
    ax.text(0.50, 0.225, "共享主干梯度 = ∂Lsupport/∂F + ∂Ldbz/∂F", ha="center", color=GREEN, fontsize=16, weight="bold")
    ax.text(0.50, 0.165, "Support头只收到Lsupport梯度；dBZ头只收到Ldbz梯度；共享层同时收到两者", ha="center", color=INK, fontsize=11)
    ax.text(0.50, 0.095, "风险：两个任务难度、梯度尺度或方向不一致时，共享特征可能互相牵制——这正是Stage 2第一版需要重新拆解的原因。", ha="center", color=RED, fontsize=10.5, weight="bold")
    save(fig, out / "figures/06_multitask_loss_gradient_flow.png", dpi=dpi)


def draw_stage1_performance(metrics: Mapping[str, Any], out: Path, dpi: int) -> None:
    overall = metrics["patch_evaluation"]["metrics"]["rain"]["all"]
    bins = metrics["patch_evaluation"]["metrics"]["rain"]["target_bins_mm_h"]
    type_metrics = metrics["patch_evaluation"]["metrics"]["precipitation_type"]
    fig, ax = slide(
        "Stage 1最终模型：真实DPR反射率 → 卫星降水率",
        "完整38轨validation、1,195,966个可靠正降水体素；检查点epoch 22。",
    )
    metric_card(ax, 0.055, 0.68, 0.17, 0.13, "Pearson r", f"{overall['pearson_r']:.3f}", "空间变化同步程度", BLUE)
    metric_card(ax, 0.245, 0.68, 0.17, 0.13, "CCC", f"{overall['ccc']:.3f}", "相关性＋数值一致性", GREEN)
    metric_card(ax, 0.435, 0.68, 0.17, 0.13, "MAE", f"{overall['mae']:.3f}", "mm/h", ORANGE)
    metric_card(ax, 0.625, 0.68, 0.17, 0.13, "RMSE", f"{overall['rmse']:.3f}", "mm/h，受强降水影响大", RED)
    metric_card(ax, 0.815, 0.68, 0.13, 0.13, "类型准确率", f"{type_metrics['accuracy']:.3f}", "T3D辅助任务", PURPLE)

    bin_order = ["lt_1", "1_to_5", "5_to_10", "10_to_30", "ge_30"]
    labels = ["<1", "1–5", "5–10", "10–30", "≥30"]
    r_values = [float(bins[k]["pearson_r"]) for k in bin_order]
    rmse = [float(bins[k]["rmse"]) for k in bin_order]
    counts = [int(bins[k]["count"]) for k in bin_order]
    plot = fig.add_axes([0.075, 0.16, 0.55, 0.41])
    x = np.arange(len(labels))
    bars = plot.bar(x - 0.18, r_values, width=0.36, color=BLUE, label="Pearson r")
    plot.set_ylim(0, 1.0)
    plot.set_ylabel("Pearson r")
    plot.set_xticks(x, labels)
    plot.set_xlabel("")
    plot.grid(axis="y", color=GRID, alpha=0.6)
    plot.spines[["top"]].set_visible(False)
    second = plot.twinx()
    second.plot(x + 0.18, rmse, color=RED, marker="o", lw=2.2, label="RMSE")
    second.set_ylabel("RMSE (mm/h)", color=RED)
    second.tick_params(axis="y", colors=RED)
    for b, v in zip(bars, r_values):
        plot.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", fontsize=9, color=NAVY)
    for xi, count in zip(x, counts):
        plot.text(xi, 0.03, f"N={count:,}", ha="center", va="bottom", fontsize=7.8, rotation=90, color=MUTED)
    handles1, labels1 = plot.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    plot.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)
    fig.text(0.35, 0.105, "目标降水强度区间 (mm/h)", ha="center", color=INK, fontsize=10.5)

    rounded_box(
        ax,
        0.68,
        0.39,
        0.27,
        0.18,
        "可以确认",
        "整体反射率→降水映射学习充分\n低于5 mm/h最稳定\n类型辅助头达到83.6%准确率",
        face="#EDF9F3",
        edge=GREEN,
    )
    rounded_box(
        ax,
        0.68,
        0.15,
        0.27,
        0.18,
        "仍需注意",
        "5–30 mm/h分段相关性偏低\n≥30 mm/h明显低估且样本极少\n输入若偏离真实DPR域，性能会迅速下降",
        face="#FFF2F0",
        edge=RED,
        title_color=RED,
    )
    save(fig, out / "figures/07_stage1_performance.png", dpi=dpi)


def draw_stage2_performance(metrics: Mapping[str, Any], static: Mapping[str, Any], out: Path, dpi: int) -> None:
    values = metrics["metrics"]
    dbz = values["reflectivity_on_target_support"]
    support = values["support"]
    fig, ax = slide(
        "Stage 2最终第一版W1.25：GR稀疏反射率 → DPR反射率与支持域",
        "完整38轨validation；dBZ检查点epoch 27；support阈值0.80由validation CSI选择。",
    )
    metric_card(ax, 0.055, 0.69, 0.17, 0.12, "dBZ Pearson r", f"{dbz['pearson_r']:.3f}", "全部DPR目标位置", BLUE)
    metric_card(ax, 0.245, 0.69, 0.17, 0.12, "dBZ RMSE", f"{dbz['rmse_dbz']:.3f}", "dBZ", PURPLE)
    metric_card(ax, 0.435, 0.69, 0.17, 0.12, "Support CSI", f"{support['csi']:.3f}", "交并比", ORANGE)
    metric_card(ax, 0.625, 0.69, 0.14, 0.12, "Recall", f"{support['recall']:.3f}", "找到多少真实回波", GREEN)
    metric_card(ax, 0.785, 0.69, 0.14, 0.12, "Precision", f"{support['precision']:.3f}", "预测回波有多少是真的", CYAN)

    region_names = ["Q11直接重叠", "gap邻近可达", "outside远离观测", "≥35 dBZ"]
    region_keys = ["q11_direct_overlap", "dpr_gap_proxy", "dpr_outside_proxy", "dpr_dbz_ge35"]
    regions = {
        str(row["region"]): row for row in static["regions"]
    }
    rmse = [float(regions[k]["rmse_dbz"]) for k in region_keys]
    corr = [float(regions[k]["pearson_r"]) for k in region_keys]
    colors_ = [GREEN, BLUE, ORANGE, RED]
    plot = fig.add_axes([0.075, 0.17, 0.56, 0.40])
    x = np.arange(len(region_names))
    bars = plot.bar(x, rmse, color=colors_, width=0.55)
    plot.set_xticks(x, region_names)
    plot.set_ylabel("RMSE (dBZ)")
    plot.set_ylim(0, max(rmse) * 1.25)
    plot.grid(axis="y", color=GRID, alpha=0.6)
    plot.spines[["top", "right"]].set_visible(False)
    for b, e, r in zip(bars, rmse, corr):
        plot.text(b.get_x() + b.get_width() / 2, e + 0.2, f"RMSE={e:.2f}\nr={r:.2f}", ha="center", fontsize=9)

    rounded_box(
        ax,
        0.68,
        0.38,
        0.27,
        0.19,
        "为什么总体r=0.702仍不够",
        "Q11区域已达到r=0.841\n但outside只有r=0.428\n强回波r=0.171且Bias=-7.37 dBZ",
        face="#FFF2F0",
        edge=RED,
        title_color=RED,
    )
    rounded_box(
        ax,
        0.68,
        0.15,
        0.27,
        0.17,
        "Support同样是瓶颈",
        "真实回波中31.6%被漏掉\n漏报进入Stage 1后会直接变成0降水\n不能只看dBZ回归指标",
        face="#FFF8EC",
        edge=ORANGE,
    )
    save(fig, out / "figures/08_stage2_performance.png", dpi=dpi)


def _rain_metrics(cascade: Mapping[str, Any], mode: str) -> Mapping[str, float]:
    return cascade["metrics"][mode]["reliable_positive"]["rain"]["all"]


def draw_cascade(cascade: Mapping[str, Any], out: Path, dpi: int) -> None:
    modes = [
        ("dpr_oracle", "真实DPR\n→Stage 1", GREEN),
        ("w1_25_true_dbz_predicted_mask", "真实dBZ＋\n预测support", CYAN),
        ("w1_25_oracle_mask", "预测dBZ＋\n真实support", PURPLE),
        ("w1_25_predicted_mask", "W1.25完整\n可部署串联", RED),
        ("gr_interp", "师兄插值GR\n→Stage 1", ORANGE),
    ]
    values = [_rain_metrics(cascade, mode) for mode, _, _ in modes]
    fig, ax = slide(
        "两阶段串联后为什么劣化：Stage 2的值场误差比support误差更致命",
        "同一38轨validation、同一Stage 1、同一评价mask；Oracle仅用于定位误差来源，不能部署。",
    )
    plot_r = fig.add_axes([0.07, 0.19, 0.42, 0.55])
    x = np.arange(len(modes))
    colors_ = [m[2] for m in modes]
    r = [float(v["pearson_r"]) for v in values]
    bars = plot_r.bar(x, r, color=colors_, width=0.62)
    plot_r.axhline(0.68, color=NAVY, ls="--", lw=1.5, label="师兄PPT约r=0.68（口径待统一）")
    plot_r.set_xticks(x, [m[1] for m in modes])
    plot_r.set_ylim(0, 1.0)
    plot_r.set_ylabel("最终降水 Pearson r")
    plot_r.grid(axis="y", color=GRID, alpha=0.6)
    plot_r.spines[["top", "right"]].set_visible(False)
    plot_r.legend(frameon=False, fontsize=8.5, loc="upper right")
    for b, v in zip(bars, r):
        plot_r.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}", ha="center", weight="bold", fontsize=9)

    plot_e = fig.add_axes([0.57, 0.42, 0.36, 0.32])
    rmse = [float(v["rmse"]) for v in values]
    bars2 = plot_e.barh(np.arange(len(modes)), rmse, color=colors_)
    plot_e.set_yticks(np.arange(len(modes)), [m[1].replace("\n", " ") for m in modes])
    plot_e.invert_yaxis()
    plot_e.set_xlabel("最终降水 RMSE (mm/h)")
    plot_e.grid(axis="x", color=GRID, alpha=0.6)
    plot_e.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars2, rmse):
        plot_e.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)

    rounded_box(
        ax,
        0.56,
        0.14,
        0.38,
        0.17,
        "2×2反事实审计给出的判断",
        "真实dBZ＋预测support仍有r=0.708；预测dBZ＋真实support只有r=0.471。\n因此当前主要矛盾是Stage 2值场/补全，而不只是support阈值。",
        face="#FFF2F0",
        edge=RED,
        title_color=RED,
        body_size=9,
    )
    save(fig, out / "figures/09_cascade_error_propagation.png", dpi=dpi)


def draw_error_budget(static: Mapping[str, Any], out: Path, dpi: int) -> None:
    regions = {
        str(row["region"]): row for row in static["regions"]
    }
    names = ["Q11\n直接重叠", "gap\n邻近可达", "outside\n远离观测"]
    keys = ["q11_direct_overlap", "dpr_gap_proxy", "dpr_outside_proxy"]
    target_share = [100 * float(regions[k]["target_fraction"]) for k in keys]
    sse_share = [100 * float(regions[k]["squared_error_fraction"]) for k in keys]
    fn_share = [100 * float(regions[k]["false_negative_fraction"]) for k in keys]
    fig, ax = slide(
        "Stage 2误差预算：三分之一的outside区域贡献了一半以上误差",
        "三类可观测区域互不重叠；强回波与它们重叠，因此强回波误差不能再相加。",
    )
    plot = fig.add_axes([0.075, 0.20, 0.56, 0.54])
    x = np.arange(3)
    width = 0.24
    plot.bar(x - width, target_share, width, color=BLUE, label="DPR目标体素占比")
    plot.bar(x, sse_share, width, color=RED, label="dBZ平方误差贡献")
    plot.bar(x + width, fn_share, width, color=ORANGE, label="support漏报贡献")
    plot.set_xticks(x, names)
    plot.set_ylabel("占比 (%)")
    plot.set_ylim(0, 90)
    plot.grid(axis="y", color=GRID, alpha=0.6)
    plot.spines[["top", "right"]].set_visible(False)
    plot.legend(frameon=False, loc="upper left")
    for xpos, values, offset in ((x - width, target_share, 0), (x, sse_share, 0), (x + width, fn_share, 0)):
        for xx, value in zip(xpos, values):
            plot.text(xx, value + 1.5, f"{value:.1f}%", ha="center", fontsize=8.5)

    strong = regions["dpr_dbz_ge35"]
    metric_card(ax, 0.69, 0.61, 0.24, 0.14, "≥35 dBZ强回波", "8.1%目标", "却贡献32.6% dBZ平方误差", RED)
    metric_card(ax, 0.69, 0.42, 0.24, 0.14, "outside区域", "r=0.428", "RMSE=6.88 dBZ；漏报贡献80.4%", ORANGE)
    local = static["local_shift_oracle"]["metrics"]
    metric_card(
        ax,
        0.69,
        0.23,
        0.24,
        0.14,
        "局部位移Oracle",
        f"CSI +{float(local['local_window_height_csi_gain']):.3f}",
        "存在局部错位，但不是唯一主因",
        PURPLE,
    )
    ax.text(0.69, 0.145, "推论：继续微调一个共享双头U-Net很难同时解决远区补全、强回波长尾和局部错位。", color=RED, fontsize=10.5, weight="bold", wrap=True)
    save(fig, out / "figures/10_stage2_error_budget.png", dpi=dpi)


def draw_findings(out: Path, dpi: int) -> None:
    fig, ax = slide(
        "目前得到的实验结论，以及需要和导师确认的下一步",
        "汇报重点应从“又试了一个模型”转向“证据如何改变任务定义”。",
    )
    findings = [
        ("1", "数据与输入", "按日期分组无泄漏；非重叠输出核心＋重叠上下文；高度不压缩", BLUE),
        ("2", "Stage 1", "真实DPR输入时r=0.864，已具备较好的数值转换能力，可暂时封版", GREEN),
        ("3", "Stage 2", "总体dBZ r=0.702看似可用，但outside、强回波和support漏报仍严重", ORANGE),
        ("4", "串联", "可部署W1.25串联r=0.454，明显低于Stage 1上限及师兄约0.68参考线", RED),
        ("5", "误差定位", "真实dBZ＋预测support为r=0.708，表明值场恢复比support阈值更关键", PURPLE),
    ]
    for i, (number, name, body, color_) in enumerate(findings):
        y = 0.75 - i * 0.125
        ax.add_patch(plt.Circle((0.075, y + 0.035), 0.027, facecolor=color_, edgecolor=WHITE))
        ax.text(0.075, y + 0.035, number, ha="center", va="center", color=WHITE, weight="bold")
        ax.text(0.12, y + 0.055, name, color=color_, fontsize=11.5, weight="bold", va="center")
        ax.text(0.12, y + 0.015, body, color=INK, fontsize=10.2, va="center")

    rounded_box(
        ax,
        0.60,
        0.58,
        0.34,
        0.22,
        "建议下一阶段：Stage 2-v2任务拆解",
        "R1 稀疏空间恢复上限\nR2 重叠区标定/有限配准\nR3 独立support恢复\nR4 条件dBZ补全与不确定性\nR5 分阶段融合后再重新串联",
        face="#EAF3FF",
        edge=BLUE,
        body_size=9.3,
    )
    rounded_box(
        ax,
        0.60,
        0.25,
        0.34,
        0.24,
        "希望导师重点决策",
        "① 是否认可先停下Stage 3，集中攻克Stage 2？\n② outside区域应追求确定值，还是同时输出不确定性？\n③ 师兄r≈0.68的评价数据和mask能否进一步统一？\n④ 能否获得GR加工流程/原始仰角信息用于后续验证？",
        face="#FFF8EC",
        edge=ORANGE,
        body_size=9.2,
    )
    ax.text(0.60, 0.135, "正在运行：S2-R0冻结Stage 1区域Oracle，将进一步量化每个子任务能关闭多少最终降水误差。", color=RED, fontsize=10.5, weight="bold", wrap=True)
    save(fig, out / "figures/11_findings_and_discussion.png", dpi=dpi)


def _safe_nanmax(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    filled = np.where(finite, values, -np.inf)
    result = np.max(filled, axis=axis)
    result[~np.any(finite, axis=axis)] = np.nan
    return result


def draw_stage2_orbit(npz_path: Path, destination: Path, dpi: int) -> None:
    with np.load(npz_path) as data:
        probability = np.asarray(data["support_probability"], dtype=np.float32)
        predicted_support = np.asarray(data["predicted_support"], dtype=bool)
        predicted_dbz = np.asarray(data["predicted_dbz"], dtype=np.float32)
        target_support = np.asarray(data["target_support"], dtype=bool)
        target_dbz = np.asarray(data["target_dbz"], dtype=np.float32)
        heights = np.asarray(data["heights_km"], dtype=np.float32)
    sample_id = npz_path.stem
    target_field = np.where(target_support, target_dbz, np.nan)
    predicted_field = np.where(predicted_support, predicted_dbz, np.nan)
    target_max = _safe_nanmax(target_field, axis=-1)
    predicted_max = _safe_nanmax(predicted_field, axis=-1)
    target_depth = target_support.sum(axis=-1)
    predicted_depth = predicted_support.sum(axis=-1)
    ray_index = int(np.argmax(target_support.sum(axis=(0, 2))))

    common = target_support & predicted_support
    if np.any(common):
        truth = target_dbz[common].astype(np.float64)
        pred = predicted_dbz[common].astype(np.float64)
        corr = float(np.corrcoef(truth, pred)[0, 1]) if truth.size > 1 else math.nan
        rmse = float(np.sqrt(np.mean(np.square(pred - truth))))
    else:
        corr = rmse = math.nan
    tp = int((target_support & predicted_support).sum())
    fp = int((~target_support & predicted_support).sum())
    fn = int((target_support & ~predicted_support).sum())
    csi = tp / (tp + fp + fn) if tp + fp + fn else math.nan

    fig = plt.figure(figsize=(13.333, 7.5), facecolor=WHITE)
    fig.suptitle("Stage 2轨道实例：DPR反射率与支持域重建", fontsize=21, weight="bold", color=NAVY, y=0.985)
    short_id = sample_id.replace("2A.GPM.DPR.V9-20211125.", "")
    fig.text(0.5, 0.942, f"{short_id} | 公共support dBZ: r={corr:.3f}, RMSE={rmse:.2f} dBZ | support CSI={csi:.3f}", ha="center", color=MUTED, fontsize=9.5)
    axes = fig.subplots(3, 2)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.075, top=0.875, hspace=0.58, wspace=0.18)
    cmap_dbz = plt.get_cmap("turbo").copy()
    cmap_dbz.set_bad("#F1F3F5")
    im0 = axes[0, 0].imshow(target_max.T, origin="lower", aspect="auto", cmap=cmap_dbz, vmin=10, vmax=50)
    axes[0, 0].set_title("真实DPR：垂直最大dBZ")
    im1 = axes[0, 1].imshow(predicted_max.T, origin="lower", aspect="auto", cmap=cmap_dbz, vmin=10, vmax=50)
    axes[0, 1].set_title("Stage 2：垂直最大dBZ")
    fig.colorbar(im1, ax=axes[0, :].tolist(), shrink=0.82, pad=0.02, label="dBZ")

    true_section = np.where(target_support[:, ray_index, :], target_dbz[:, ray_index, :], np.nan)
    pred_section = np.where(predicted_support[:, ray_index, :], predicted_dbz[:, ray_index, :], np.nan)
    extent = [0, target_support.shape[0], float(heights[0]), float(heights[-1])]
    axes[1, 0].imshow(true_section.T, origin="lower", aspect="auto", extent=extent, cmap=cmap_dbz, vmin=10, vmax=50)
    axes[1, 0].set_title(f"真实DPR垂直剖面（ray={ray_index}）")
    axes[1, 1].imshow(pred_section.T, origin="lower", aspect="auto", extent=extent, cmap=cmap_dbz, vmin=10, vmax=50)
    axes[1, 1].set_title(f"Stage 2垂直剖面（ray={ray_index}）")
    for a in axes[1, :]:
        a.set_ylabel("高度 (km)")
        a.set_xlabel("scan")

    im4 = axes[2, 0].imshow(target_depth.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=30)
    axes[2, 0].set_title("真实DPR：每个廓线的回波层数")
    im5 = axes[2, 1].imshow(predicted_depth.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=30)
    axes[2, 1].set_title("Stage 2：每个廓线的预测回波层数")
    fig.colorbar(im5, ax=axes[2, :].tolist(), shrink=0.82, pad=0.02, label="有效高度层数")
    for row in (0, 2):
        for a in axes[row, :]:
            if row == 2:
                a.set_xlabel("scan")
            a.set_ylabel("ray")
    save(fig, destination, dpi=dpi)


def copy_existing_assets(out: Path) -> list[dict[str, str]]:
    selected: list[tuple[str, str, str]] = [
        (
            "outputs/ablations/stage1_i_g002_t3d/analysis/test_predictions/sample_03/diagnostics.png",
            "orbits/stage1_representative_wet_track.png",
            "Stage 1较长、降水样本较丰富的测试轨道诊断",
        ),
        (
            "outputs/ablations/stage1_i_g002_t3d/analysis/test_predictions/sample_05/diagnostics.png",
            "orbits/stage1_heavy_rain_track.png",
            "Stage 1含强降水和低层误差的测试轨道诊断",
        ),
        (
            "outputs/ablations/stage1_i_g002_t3d/analysis/test_predictions/aggregate_diagnostics.png",
            "orbits/stage1_six_track_aggregate.png",
            "Stage 1固定六轨总体分布、相关性及高度误差",
        ),
        (
            "outputs/stage3_c0_2x2/validation_d_vs_w1p25/visualizations/2A.GPM.DPR.V9-20211125.20170503-S070958-E084232.018057.V07A/comparisons/00_all_methods_overview.png",
            "orbits/cascade_long_track_overview.png",
            "长轨道上真实DPR、插值GR、D/W1.25及Oracle串联结果总览",
        ),
        (
            "outputs/stage3_c0_2x2/validation_d_vs_w1p25/visualizations/2A.GPM.DPR.V9-20211125.20170805-S035108-E052342.019517.V07A/comparisons/00_all_methods_overview.png",
            "orbits/cascade_wet_track_overview.png",
            "较强降水轨道的所有串联方法总览",
        ),
        (
            "outputs/stage3_c0_2x2/validation_d_vs_w1p25/visualizations/2A.GPM.DPR.V9-20211125.20170503-S070958-E084232.018057.V07A/comparisons/08_w1_25_predicted_mask_vs_pre_dpr.png",
            "orbits/cascade_w1p25_failure_detail.png",
            "W1.25可部署串联与卫星降水的详细剖面/分布对比",
        ),
        (
            "outputs/ablations/stage1_i_g002_t3d/analysis/training_history/training_overview.png",
            "training/stage1_training_overview.png",
            "Stage 1训练/验证损失和指标随epoch变化",
        ),
        (
            "outputs/stage2_r0_decomposition_audit/static/plots/region_error_budget.png",
            "training/stage2_r0_region_error_budget_original.png",
            "R0静态区域误差预算原始图",
        ),
    ]
    records: list[dict[str, str]] = []
    for source_rel, destination_rel, description in selected:
        source = PROJECT_ROOT / source_rel
        destination = out / destination_rel
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "output": destination_rel,
                "source": source_rel,
                "description": description,
            }
        )
    return records


def write_manifest(out: Path, generated: Sequence[dict[str, str]], copied: Sequence[dict[str, str]]) -> None:
    lines = [
        "# 降水反演进展汇报图片索引",
        "",
        "> 本目录由 `scripts/build_progress_report_assets.py` 在CPU上生成。建议所有图片在PPT中保持原始宽高比。",
        "",
        "## 新生成的技术图",
        "",
    ]
    for record in generated:
        lines.extend(
            [
                f"### `{record['output']}`",
                "",
                record["description"],
                "",
                f"![{record['output']}]({record['output']})",
                "",
            ]
        )
    lines.extend(["## 集中的已有实验图", ""])
    for record in copied:
        lines.extend(
            [
                f"### `{record['output']}`",
                "",
                record["description"],
                "",
                f"原始来源：`{record['source']}`",
                "",
                f"![{record['output']}]({record['output']})",
                "",
            ]
        )
    (out / "figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    out = args.output_dir.expanduser().resolve()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is non-empty; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    split = load_json("metadata/splits/split_summary.json")
    indices = {
        key: load_json(f"metadata/stage1_patch_indices/{key}.json")
        for key in ("train", "val", "test")
    }
    stage1 = load_json("outputs/ablations/stage1_i_g002_t3d/analysis/full_validation/metrics.json")
    stage2 = load_json(
        "outputs/stage2_ablations/four_channel_distance_intensity_w1p25/analysis/validation_candidates/reflectivity/metrics.json"
    )
    static = load_json("outputs/stage2_r0_decomposition_audit/static/summary.json")
    cascade = load_json("outputs/stage3_c0_2x2/validation_d_vs_w1p25/metrics.json")

    draw_overall_route(out, args.dpi)
    draw_split_and_patch(split, indices, out, args.dpi)
    draw_stage1_architecture(out, args.dpi)
    draw_stage2_architecture(out, args.dpi)
    draw_masks(out, args.dpi)
    draw_loss_flow(out, args.dpi)
    draw_stage1_performance(stage1, out, args.dpi)
    draw_stage2_performance(stage2, static, out, args.dpi)
    draw_cascade(cascade, out, args.dpi)
    draw_error_budget(static, out, args.dpi)
    draw_findings(out, args.dpi)

    stage2_orbits = sorted(
        (
            PROJECT_ROOT
            / "outputs/stage2_ablations/four_channel_distance_intensity_w1p25/analysis/validation_candidates/reflectivity/orbits"
        ).glob("*.npz")
    )
    if len(stage2_orbits) < 2:
        raise FileNotFoundError("expected two saved W1.25 validation orbit files")
    orbit_records: list[dict[str, str]] = []
    for position, source in enumerate(stage2_orbits[:2], start=1):
        destination_rel = f"orbits/stage2_orbit_{position:02d}_{source.stem}.png"
        draw_stage2_orbit(source, out / destination_rel, args.dpi)
        orbit_records.append(
            {
                "output": destination_rel,
                "source": str(source.relative_to(PROJECT_ROOT)),
                "description": "W1.25验证轨道的目标/预测dBZ、垂直剖面和support深度对比",
            }
        )

    generated = [
        {"output": "figures/01_overall_task_route.png", "description": "总体两阶段任务与监督来源"},
        {"output": "figures/02_dataset_split_and_patch.png", "description": "轨道划分、滑窗、padding与Batch形状"},
        {"output": "figures/03_stage1_architecture.png", "description": "Stage 1高度保持3D U-Net和T3D辅助头"},
        {"output": "figures/04_stage2_architecture.png", "description": "Stage 2共享骨干和双输出头"},
        {"output": "figures/05_masks_and_supervision.png", "description": "主要mask的判定条件与用途"},
        {"output": "figures/06_multitask_loss_gradient_flow.png", "description": "双任务损失相加与梯度回传"},
        {"output": "figures/07_stage1_performance.png", "description": "Stage 1完整验证集性能与强度分层"},
        {"output": "figures/08_stage2_performance.png", "description": "Stage 2支持域、dBZ及分区性能"},
        {"output": "figures/09_cascade_error_propagation.png", "description": "2×2反事实串联审计与误差传播"},
        {"output": "figures/10_stage2_error_budget.png", "description": "Stage 2可观测区域和强回波误差预算"},
        {"output": "figures/11_findings_and_discussion.png", "description": "当前结论、下一路线和导师决策问题"},
        *orbit_records,
    ]
    copied = copy_existing_assets(out)
    provenance = {
        "format": "precipitation_inversion_progress_report_assets_v1",
        "gpu_used": False,
        "generated_outputs": generated,
        "copied_outputs": copied,
        "metric_sources": [
            "metadata/splits/split_summary.json",
            "outputs/ablations/stage1_i_g002_t3d/analysis/full_validation/metrics.json",
            "outputs/stage2_ablations/four_channel_distance_intensity_w1p25/analysis/validation_candidates/reflectivity/metrics.json",
            "outputs/stage3_c0_2x2/validation_d_vs_w1p25/metrics.json",
            "outputs/stage2_r0_decomposition_audit/static/summary.json",
        ],
    }
    (out / "figure_sources.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(out, generated, copied)
    print(f"Progress-report assets -> {out}")


if __name__ == "__main__":
    main()
