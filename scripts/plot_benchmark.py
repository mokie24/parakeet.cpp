#!/usr/bin/env python3
"""
plot_benchmark.py — Generate benchmark plots from parakeet.cpp JSON results.

Usage:
    python scripts/plot_benchmark.py --results benchmarks/results --out benchmarks/plots
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Style ──────────────────────────────────────────────────────────────────────

COLORS = {
    "nemo":   "#4e79a7",  # blue  – NeMo / PyTorch
    "f32":    "#f28e2b",  # orange – ours f32
    "q8_0":   "#59a14f",  # green  – ours q8_0
}

HATCHES = {
    "nemo":  "",
    "f32":   "//",
    "q8_0":  "xx",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.4,
    "font.size": 10,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict]:
    """Load all <model>.json files (skip threads.json) and return sorted list."""
    models = []
    for p in sorted(results_dir.glob("*.json")):
        if p.stem == "threads":
            continue
        with open(p) as f:
            models.append(json.load(f))
    return models


def load_threads(results_dir: Path) -> list[dict] | None:
    """Load threads.json if present."""
    p = results_dir / "threads.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        # The runner writes {"model",...,"sweep":[{threads,nemo,ours}, ...]};
        # accept either that or a bare list.
        return d["sweep"] if isinstance(d, dict) and "sweep" in d else d
    return None


def _short(name: str) -> str:
    """Shorten model name for axis labels."""
    return name.replace("parakeet-", "").replace("parakeet_realtime_eou_", "rt-eou-")


# ── Per-chart helpers ──────────────────────────────────────────────────────────

def _grouped_bar(ax, labels, groups: dict[str, list[float]], ylabel: str, title: str):
    """Draw grouped bars.  groups = {series_label: [values]}."""
    n_models = len(labels)
    n_series = len(groups)
    width = 0.7 / n_series
    x = np.arange(n_models)
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    for (label, values), offset in zip(groups.items(), offsets):
        key = label.split()[0].lower().replace("-", "_")  # "nemo" / "f32" / "q8_0"
        color = COLORS.get(key, "#888888")
        hatch = HATCHES.get(key, "")
        bars = ax.bar(
            x + offset, values, width,
            label=label, color=color, hatch=hatch, edgecolor="white", linewidth=0.5,
        )
        # Annotate bars if few models
        if n_models <= 4:
            for bar, v in zip(bars, values):
                if v and not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels([_short(l) for l in labels], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)


# ── Individual plots ───────────────────────────────────────────────────────────

def plot_rtfx(models: list[dict], out_dir: Path):
    labels = [m["model"] for m in models]
    nemo_vals, f32_vals, q8_vals = [], [], []
    for m in models:
        ls = m["manifests"]["librispeech"]
        nemo_vals.append(ls["nemo"]["rtfx"])
        f32_vals.append(ls["ours"]["f32"]["rtfx"])
        q8_vals.append(ls["ours"]["q8_0"]["rtfx"])

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    _grouped_bar(ax, labels, {
        "NeMo (PyTorch CPU)": nemo_vals,
        "ours f32":           f32_vals,
        "ours q8_0":          q8_vals,
    }, ylabel="RTFx  (audio_sec / proc_sec)  — higher = faster",
       title="Real-Time Factor × — NeMo vs parakeet.cpp  (LibriSpeech test-clean)")

    ax.axhline(1.0, color="red", linewidth=1.2, linestyle="--", alpha=0.7, label=">1 = real-time")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.text(0.01, 0.97, ">1 = faster than real-time", transform=ax.transAxes,
            fontsize=8, va="top", color="red", alpha=0.75)

    fig.tight_layout()
    fig.savefig(out_dir / "rtfx.png")
    plt.close(fig)
    print(f"  wrote rtfx.png  (models={len(labels)})")


def plot_speedup(models: list[dict], out_dir: Path):
    labels = [m["model"] for m in models]
    f32_su, q8_su = [], []
    for m in models:
        ls = m["manifests"]["librispeech"]
        nemo_rtfx = ls["nemo"]["rtfx"]
        f32_su.append(ls["ours"]["f32"]["rtfx"] / nemo_rtfx)
        q8_su.append(ls["ours"]["q8_0"]["rtfx"] / nemo_rtfx)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    _grouped_bar(ax, labels, {
        "f32":  f32_su,
        "q8_0": q8_su,
    }, ylabel="Speedup  (ours RTFx / NeMo RTFx)",
       title="Speedup vs NeMo  (ours / NeMo RTFx)  — >1 = ours is faster")

    ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", alpha=0.8, label="parity (=NeMo)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.text(0.01, 0.97, "<1 = slower than NeMo; >1 = faster than NeMo",
            transform=ax.transAxes, fontsize=8, va="top", color="dimgray")

    fig.tight_layout()
    fig.savefig(out_dir / "speedup.png")
    plt.close(fig)
    print(f"  wrote speedup.png")


def plot_wer(models: list[dict], out_dir: Path):
    labels = [m["model"] for m in models]
    nemo_w, f32_w, q8_w = [], [], []
    for m in models:
        ls = m["manifests"]["librispeech"]
        nemo_w.append(ls["nemo"]["wer_vs_truth"] * 100)
        f32_w.append(ls["ours"]["f32"]["wer_vs_truth"] * 100)
        q8_w.append(ls["ours"]["q8_0"]["wer_vs_truth"] * 100)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    _grouped_bar(ax, labels, {
        "NeMo": nemo_w,
        "f32":  f32_w,
        "q8_0": q8_w,
    }, ylabel="WER vs ground truth  (%)",
       title="WER vs LibriSpeech test-clean ground truth  (lower = better)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "wer.png")
    plt.close(fig)
    print(f"  wrote wer.png")


def plot_agreement(models: list[dict], out_dir: Path):
    labels = [m["model"] for m in models]
    f32_a, q8_a = [], []
    for m in models:
        ls = m["manifests"]["librispeech"]
        f32_a.append(ls["ours"]["f32"]["agreement_wer_vs_nemo"] * 100)
        q8_a.append(ls["ours"]["q8_0"]["agreement_wer_vs_nemo"] * 100)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    _grouped_bar(ax, labels, {
        "f32":  f32_a,
        "q8_0": q8_a,
    }, ylabel="Agreement WER (ours vs NeMo)  (%)",
       title="Transcript Agreement  (ours vs NeMo, lower = closer match)")
    ax.set_ylim(bottom=0)
    # Use log scale if all values fit (helps when near-zero)
    max_val = max(f32_a + q8_a, default=0)
    if max_val > 0 and max_val < 1.0:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_ylabel("Agreement WER (ours vs NeMo)  (%)  [log scale]")
    ax.text(0.01, 0.97, "~0% = our transcripts match NeMo output",
            transform=ax.transAxes, fontsize=8, va="top", color="dimgray")

    fig.tight_layout()
    fig.savefig(out_dir / "agreement.png")
    plt.close(fig)
    print(f"  wrote agreement.png")


def plot_latency_vs_len(models: list[dict], out_dir: Path):
    """Scatter: per-file proc time vs audio length, pooled across all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    nemo_x, nemo_y = [], []
    f32_x,  f32_y  = [], []
    q8_x,   q8_y   = [], []

    for m in models:
        for mname, mdata in m["manifests"].items():
            # NeMo: proc_s
            for f in mdata["nemo"]["files"]:
                nemo_x.append(f["audio_sec"])
                nemo_y.append(f["proc_s"])
            # Ours f32/q8_0: proc_s (ours stores proc_s since runner converts from proc_ms)
            for dtype, ddata in mdata["ours"].items():
                for f in ddata["files"]:
                    proc_s = f.get("proc_s") or f.get("proc_ms", 0) / 1000.0
                    if dtype == "f32":
                        f32_x.append(f["audio_sec"])
                        f32_y.append(proc_s)
                    else:
                        q8_x.append(f["audio_sec"])
                        q8_y.append(proc_s)

    kw = dict(alpha=0.5, s=18, edgecolors="none")
    if nemo_x:
        ax.scatter(nemo_x, nemo_y, color=COLORS["nemo"],  label="NeMo",      **kw)
    if f32_x:
        ax.scatter(f32_x,  f32_y,  color=COLORS["f32"],   label="ours f32",  **kw)
    if q8_x:
        ax.scatter(q8_x,   q8_y,   color=COLORS["q8_0"],  label="ours q8_0", **kw)

    # Real-time line: proc_s = audio_sec
    if nemo_x or f32_x:
        all_x = (nemo_x + f32_x + q8_x)
        xmax = max(all_x) * 1.05
        xs = np.linspace(0, xmax, 100)
        ax.plot(xs, xs, "r--", linewidth=1.2, alpha=0.7, label="real-time (1×)")

    ax.set_xlabel("Audio length  (s)")
    ax.set_ylabel("Processing time  (s)")
    ax.set_title("Latency vs Audio Length  (per file, all models & clips)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_len.png")
    plt.close(fig)
    print(f"  wrote latency_vs_len.png  (nemo={len(nemo_x)} pts, f32={len(f32_x)}, q8_0={len(q8_x)})")


def plot_memory(models: list[dict], out_dir: Path):
    labels = [m["model"] for m in models]
    nemo_m, f32_m, q8_m = [], [], []
    for m in models:
        ls = m["manifests"]["librispeech"]
        nemo_m.append(ls["nemo"]["peak_rss_mb"])
        f32_m.append(ls["ours"]["f32"]["peak_rss_mb"])
        q8_m.append(ls["ours"]["q8_0"]["peak_rss_mb"])

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    _grouped_bar(ax, labels, {
        "NeMo": nemo_m,
        "f32":  f32_m,
        "q8_0": q8_m,
    }, ylabel="Peak RSS  (MB)",
       title="Peak Memory Usage — NeMo vs parakeet.cpp  (LibriSpeech run)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "memory.png")
    plt.close(fig)
    print(f"  wrote memory.png")


def plot_size(models: list[dict], out_dir: Path):
    labels = [m["model"] for m in models]
    f32_s, q8_s = [], []
    for m in models:
        f32_s.append(m["gguf_size_mb"]["f32"])
        q8_s.append(m["gguf_size_mb"]["q8_0"])

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    _grouped_bar(ax, labels, {
        "f32":  f32_s,
        "q8_0": q8_s,
    }, ylabel="GGUF file size  (MB)",
       title="GGUF Model Size — f32 vs q8_0  (quantized)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "size.png")
    plt.close(fig)
    print(f"  wrote size.png")


def plot_threads(threads_data: list[dict], out_dir: Path):
    threads = [row["threads"] for row in threads_data]
    nemo_rtfx = [row["nemo"]["rtfx"] for row in threads_data]
    ours_rtfx = [row["ours"]["rtfx"] for row in threads_data]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(threads, nemo_rtfx, "o-", color=COLORS["nemo"],  label="NeMo (PyTorch CPU)", linewidth=2)
    ax.plot(threads, ours_rtfx, "s-", color=COLORS["f32"],   label="ours",               linewidth=2)

    ax.set_xlabel("Thread count")
    ax.set_ylabel("RTFx  (audio_sec / proc_sec)")
    ax.set_title("Thread Scaling — RTFx vs Threads")
    ax.axhline(1.0, color="red", linewidth=1.0, linestyle="--", alpha=0.7, label="real-time")
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xticks(threads)

    fig.tight_layout()
    fig.savefig(out_dir / "threads.png")
    plt.close(fig)
    print(f"  wrote threads.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate benchmark plots from parakeet.cpp results.")
    ap.add_argument("--results", default="benchmarks/results", help="Directory with <model>.json files")
    ap.add_argument("--out", default="benchmarks/plots", help="Output directory for PNG files")
    args = ap.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_results(results_dir)
    if not models:
        print(f"ERROR: no model JSON files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(models)} model(s): {[m['model'] for m in models]}")
    print(f"Output → {out_dir}/")

    plot_rtfx(models, out_dir)
    plot_speedup(models, out_dir)
    plot_wer(models, out_dir)
    plot_agreement(models, out_dir)
    plot_latency_vs_len(models, out_dir)
    plot_memory(models, out_dir)
    plot_size(models, out_dir)

    threads_data = load_threads(results_dir)
    if threads_data:
        plot_threads(threads_data, out_dir)
    else:
        print("  skipped threads.png  (no threads.json found)")

    print("Done.")


if __name__ == "__main__":
    main()
