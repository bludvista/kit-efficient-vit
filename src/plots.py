"""All figures in plots/, drawn only from CSV files in results/.

Colours: validated categorical palette (dataviz reference palette, slots 1-5, light
mode); every series also has its own marker shape and a legend, so identity never
relies on colour alone. Skipped configurations are not drawn; they are named in a
note inside the figure.
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from common import PLOTS_DIR, RESULTS_DIR

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RES = RESULTS_DIR / "resolution"
ATT = RESULTS_DIR / "attention"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
})

# Imagenette families: (label, colour, marker)
FAMILIES = {
    "fp32": ("FP32", COLORS[0], "s"),
    "int8": ("INT8", COLORS[1], "^"),
    "tome": ("ToMe, r = 0 ... 24 (r written below the point)", COLORS[2], "o"),
    "tome_int8": ("ToMe + INT8, r = 4 ... 24 (r grows right to left)", COLORS[3], "D"),
}
# Resolution / attention methods: (label, colour, marker)
RES_METHODS = {
    "fp32": ("FP32 (SDPA attention)", COLORS[0], "s"),
    "int8": ("INT8 (SDPA attention)", COLORS[1], "^"),
    "tome_r0": ("ToMe code, no merging (explicit attention)", COLORS[2], "o"),
    "tome_p8": ("ToMe, r scaled to tokens (tome_p8)", COLORS[3], "D"),
    "tome_p8_int8": ("ToMe + INT8 (tome_p8_int8)", COLORS[4], "v"),
}
ATT_METHODS = {"naive": ("naive softmax(QK^T/sqrt(d))V", COLORS[2], "o"),
               "sdpa": ("F.scaled_dot_product_attention", COLORS[0], "s")}

plain = FuncFormatter(lambda v, _: f"{v:g}")


def family(method: str) -> tuple[str, int | None]:
    m = re.fullmatch(r"tome_r(\d+)(_int8)?", method)
    if m:
        return ("tome_int8" if m.group(2) else "tome"), int(m.group(1))
    return method, None


def save(fig, name: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plots/{name}")


def note(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)


def pareto(points: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Non-dominated points: no other point has lower x and higher-or-equal y."""
    p = points.sort_values([x, y], ascending=[True, False])
    best, keep = -1.0, []
    for i, r in p.iterrows():
        if r[y] > best:
            keep.append(i)
            best = r[y]
    return p.loc[keep]


def imagenette(xcol: str, xlabel: str, fname: str, df: pd.DataFrame, extra_note: str,
               better: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, ycol, title in [(axes[0], "top1_restricted", "Top-1, restricted to the 10 Imagenette classes"),
                            (axes[1], "top1_unrestricted", "Top-1 over all 1000 ImageNet classes")]:
        d = df.assign(acc=df[ycol] * 100)
        front = pareto(d, xcol, "acc")
        ax.scatter(front[xcol], front.acc, s=170, facecolors="none", edgecolors=INK, linewidths=1.2,
                   zorder=4, label=f"On the Pareto front (no other point is both {better} and more accurate)")
        for fam, (label, color, marker) in FAMILIES.items():
            f = d[d.family == fam].sort_values("r")
            ax.plot(f[xcol], f.acc, marker=marker, color=color, label=label,
                    lw=1.5 if len(f) > 1 else 0, markeredgecolor="white", markeredgewidth=1, zorder=3)
            if fam == "tome":  # ToMe + INT8 follows the same r order (4 ... 24, right to left)
                for _, r in f.iterrows():
                    ax.annotate(f"{int(r.r)}", (r[xcol], r.acc), xytext=(0, -14),
                                textcoords="offset points", fontsize=8, color=COLORS[2], ha="center")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_title(title, loc="left")
    fig.suptitle("DeiT-Tiny on Imagenette-160 val (3925 images, upscaled to 224 px), batch size 1, 4 threads",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, fontsize=8.5)
    fig.text(0.01, -0.14, extra_note, ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)
    save(fig, fname)


def imagenette_plots() -> None:
    acc = pd.read_csv(RESULTS_DIR / "accuracy.csv")
    lat = pd.read_csv(RESULTS_DIR / "latency.csv")[["method", "latency_median_ms"]]
    mem = pd.read_csv(RESULTS_DIR / "memory.csv")[["method", "peak_total_mb"]]
    df = acc.merge(lat, on="method").merge(mem, on="method")
    df[["family", "r"]] = df.method.apply(lambda m: pd.Series(family(m)))
    imagenette("latency_median_ms", "Latency, median (ms per image)", "imagenette_accuracy_vs_latency.png", df,
               "Latency: run 2026-10-15T14-45-58 (results/latency.csv), median of 3 interleaved blocks of 50 runs. "
               "The FP32 baseline blocks in that run spread by 7.3% (rolling median), so latency differences "
               "of a few percent are not resolved. ToMe uses explicit attention, FP32/INT8 use fused SDPA.",
               "faster")
    imagenette("peak_total_mb", "Peak memory of the process (MB, includes 305-307 MB Python/torch runtime)",
               "imagenette_accuracy_vs_memory.png", df,
               "Peak memory: results/memory.csv, fresh process that loads the saved model and runs 20 passes; "
               "one measurement per configuration (run-to-run variability not measured).", "lighter on memory")


def res_frame(name: str) -> pd.DataFrame:
    d = pd.read_csv(RES / name)
    d["tokens"] = d.n_patches + 1
    return d


def res_note(lat: pd.DataFrame) -> str:
    skipped = lat[lat.status != "ok"]
    first = skipped.sort_values("tokens").groupby("method").first()
    parts = [f"{RES_METHODS[m][0].split(' (')[0]}: from {int(r.tokens)} tokens" for m, r in first.iterrows()]
    return "Not run (estimated memory > 70% of free RAM): " + "; ".join(parts) + "."


def resolution_plots() -> None:
    lat, mem, agr = res_frame("latency.csv"), res_frame("memory.csv"), res_frame("agreement.csv")
    native = 368 * 1216 // 256 + 1
    title = "DeiT-Tiny on KITTI frames at increasing input size (aspect ratio kept), batch size 1"

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m, (label, color, marker) in RES_METHODS.items():
        d = lat[(lat.method == m) & (lat.status == "ok")].sort_values("tokens")
        lo = d[["run1_latency_median_ms", "run2_latency_median_ms"]].min(axis=1)
        hi = d[["run1_latency_median_ms", "run2_latency_median_ms"]].max(axis=1)
        ax.errorbar(d.tokens, d.latency_median_ms, yerr=[d.latency_median_ms - lo, hi - d.latency_median_ms],
                    marker=marker, color=color, label=label, capsize=3, elinewidth=1,
                    markeredgecolor="white", markeredgewidth=1)
    ax.axvline(native, color=MUTED, lw=1, ls=":")
    ax.annotate("368x1216 ≈ original KITTI size;\nlarger inputs are upscaled", (native, 0.97),
                xycoords=("data", "axes fraction"), xytext=(-5, 0), textcoords="offset points",
                ha="right", va="top", fontsize=8.5, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plain); ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel("Tokens entering the first block (patches + class token, log scale)")
    ax.set_ylabel("Latency, median (ms per image, log scale)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    note(fig, "Points: mean of the medians of two latency runs (= their median); bars: the two runs' values. "
              "Both runs failed the 10% drift rule (reference noise 16.1% / 17.8%) and were accepted together "
              "as a post-hoc decision (README). FP32 at 13569 tokens: one run only (30.2 s > 30 s limit in "
              "run 2). " + res_note(lat))
    save(fig, "resolution_latency_vs_tokens.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for m, (label, color, marker) in RES_METHODS.items():
        d = mem[(mem.method == m) & (mem.status == "ok")].sort_values("tokens")
        kw = dict(marker=marker, color=color, label=label, markeredgecolor="white", markeredgewidth=1)
        axes[0].plot(d.tokens, d.peak_total_mb, **kw)
        axes[1].plot(d.tokens, d.inference_increment_mb, **kw)
    for ax in axes:
        ax.set_xscale("log"); ax.xaxis.set_major_formatter(plain)
        ax.axvline(native, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("Tokens entering the first block (log scale)")
    axes[1].set_yscale("log"); axes[1].yaxis.set_major_formatter(plain)
    axes[0].set_ylabel("Peak memory of the process (MB)")
    axes[0].set_title("Peak memory (includes 305-307 MB runtime)", loc="left")
    axes[1].set_ylabel("Added by inference (MB, log scale)")
    axes[1].set_title("Memory added by inference (peak after - before)", loc="left")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    note(fig, "results/resolution/memory.csv: fresh process loads the saved model and runs 3 passes; one "
              "measurement per configuration. Dotted line: 368x1216 (≈ original KITTI size). " + res_note(lat))
    fig.tight_layout()
    save(fig, "resolution_memory_vs_tokens.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for m, (label, color, marker) in RES_METHODS.items():
        if m == "fp32":
            continue
        d = agr[(agr.method == m) & (agr.status == "ok")].sort_values("tokens")
        ax.plot(d.tokens, d.n_agree_with_fp32, marker=marker, color=color, label=label,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log"); ax.xaxis.set_major_formatter(plain)
    ax.set_ylim(0, 28)
    ax.set_xlabel("Tokens entering the first block (log scale)")
    ax.set_ylabel("Frames with the same top-1 as FP32 (of 27)")
    ax.set_title("Top-1 agreement with FP32 at the same input size (stability proxy, NOT accuracy)", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    skipped = agr[(agr.status != "ok") & (agr.method != "fp32")]
    note(fig, "27 frames (every 4th) of one 11-s KITTI drive; consecutive frames are highly correlated, so this "
              "is weak evidence. KITTI has no ImageNet labels. Not run: "
              + "; ".join(f"{r.method} at {r.input_size}" for _, r in skipped[skipped.tokens < 6993].iterrows())
              + " (RAM rule); all ToMe variants from 6993 tokens (RAM rule); FP32/INT8 at 13569 tokens "
              "(time budget).")
    save(fig, "resolution_agreement_vs_tokens.png")


def attention_plots() -> None:
    t, m = pd.read_csv(ATT / "timing.csv"), pd.read_csv(ATT / "memory.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for meth, (label, color, marker) in ATT_METHODS.items():
        kw = dict(marker=marker, color=color, label=label, markeredgecolor="white", markeredgewidth=1)
        d = t[(t.method == meth) & (t.status == "ok")]
        axes[0].plot(d.seq_len, d.latency_median_ms, **kw)
        d = m[(m.method == meth) & (m.status == "ok")]
        axes[1].plot(d.seq_len, d.call_increment_mb, **kw)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(plain); ax.yaxis.set_major_formatter(plain)
        ax.set_xlabel("Sequence length N (log scale)")
    axes[0].set_ylabel("Time per call, median (ms, log scale)")
    axes[0].set_title("Time", loc="left")
    axes[1].set_ylabel("Memory added by one call (MB, log scale)")
    axes[1].set_title("Memory (fresh process per point)", loc="left")
    axes[0].legend(loc="upper left", fontsize=8.5)
    drift = t.reference_drift_spread.iloc[0]
    fig.suptitle("Attention on CPU, random float32 inputs: batch 1, 3 heads, head dim 64",
                 x=0.01, ha="left", fontsize=11.5)
    note(fig, f"The timing run failed the 10% drift rule (reference spread {drift:.1%}); reported anyway, "
              "as decided before the run. Not run: naive from N = 8192 (estimated memory > 70% of free RAM); "
              "naive memory at N = 4096 (same rule, less free RAM at that time); SDPA at N = 65536 "
              "(first call 48.6 s > 30 s). On this CPU SDPA runs the flash-attention CPU kernel. "
              "CPU results only; GPU results were not measured and may differ.")
    fig.tight_layout()
    save(fig, "attention_naive_vs_sdpa.png")


if __name__ == "__main__":
    imagenette_plots()
    resolution_plots()
    attention_plots()
