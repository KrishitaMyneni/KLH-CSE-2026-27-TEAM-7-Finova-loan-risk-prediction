"""
Generate the EDA figures for the review presentation.

    python scripts/make_eda_charts.py --data data/sample_loans.csv

Writes PNGs to artifacts/figures/. Six charts, each answering one question an
examiner is likely to ask, rather than six charts for the sake of having six.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
from cleaning import clean
from data_loader import load_raw
from features import build_features

INK = "#12263F"
SLATE = "#5A7184"
HIGH = "#B23A2F"
LOW = "#1B7F5A"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 10,
        "axes.edgecolor": SLATE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)


def save(fig, name, outdir):
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()

    outdir = config.ARTIFACT_DIR / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(path=args.data, nrows=args.nrows)
    missing_before = raw.isna().mean().sort_values(ascending=False)

    df, _ = clean(raw, verbose=False)
    df = build_features(df, verbose=False)
    target = config.TARGET
    overall = df[target].mean()

    # 1 — missingness before cleaning
    top_missing = missing_before[missing_before > 0].head(15)
    if len(top_missing):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(top_missing.index[::-1], 100 * top_missing.values[::-1], color=SLATE)
        ax.set_xlabel("% of values missing")
        ax.set_title("Incomplete columns in the raw file")
        save(fig, "01_missingness.png", outdir)

    # 2 — default rate by grade
    if "grade" in df.columns:
        by_grade = df.groupby("grade", observed=True)[target].agg(["mean", "size"])
        by_grade = by_grade.sort_index()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(by_grade.index, 100 * by_grade["mean"], color=INK)
        ax.axhline(100 * overall, color=HIGH, lw=1.2, ls="--")
        ax.text(
            len(by_grade) - 0.4, 100 * overall + 0.6,
            f"book average {overall:.1%}", color=HIGH, fontsize=9, ha="right",
        )
        ax.set_ylabel("Default rate (%)")
        ax.set_title("Default rate rises monotonically with loan grade")
        save(fig, "02_default_by_grade.png", outdir)

    # 3 — default rate by purpose
    if "purpose" in df.columns:
        by_p = (
            df.groupby("purpose", observed=True)[target]
            .agg(["mean", "size"])
            .query("size >= 100")
            .sort_values("mean")
        )
        fig, ax = plt.subplots(figsize=(7, 4.6))
        colors = [HIGH if v > overall else INK for v in by_p["mean"]]
        ax.barh(by_p.index.str.replace("_", " "), 100 * by_p["mean"], color=colors)
        ax.axvline(100 * overall, color=SLATE, lw=1, ls="--")
        ax.set_xlabel("Default rate (%)")
        ax.set_title("Why the money is borrowed matters")
        save(fig, "03_default_by_purpose.png", outdir)

    # 4 — income distribution, log scale
    if "annual_inc" in df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        data = df["annual_inc"][df["annual_inc"] > 0]
        ax.hist(np.log10(data), bins=60, color=INK, alpha=0.85)
        ax.set_xlabel("Annual income (log10 scale)")
        ax.set_ylabel("Borrowers")
        ax.set_title("Income is heavily right-skewed, hence the log transform")
        save(fig, "04_income_distribution.png", outdir)

    # 5 — default rate across DTI bands
    if "dti" in df.columns:
        bands = pd.cut(df["dti"], bins=[0, 10, 15, 20, 25, 30, 40, 100])
        by_dti = df.groupby(bands, observed=True)[target].agg(["mean", "size"])
        by_dti = by_dti[by_dti["size"] >= 50]
        fig, ax = plt.subplots(figsize=(6.8, 4))
        ax.plot(
            [str(i) for i in by_dti.index], 100 * by_dti["mean"],
            marker="o", color=HIGH, lw=1.8,
        )
        ax.set_ylabel("Default rate (%)")
        ax.set_xlabel("Debt-to-income ratio")
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Higher existing debt, higher default rate")
        save(fig, "05_default_by_dti.png", outdir)

    # 6 — term and verification interaction
    if {"term", "verification_status"} <= set(df.columns):
        pivot = df.pivot_table(
            index="term", columns="verification_status", values=target, aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(6.8, 4))
        pivot.plot(kind="bar", ax=ax, color=[INK, SLATE, HIGH], width=0.75)
        ax.set_ylabel("Default rate")
        ax.set_xlabel("Term (months)")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="", frameon=False, fontsize=9)
        ax.set_title("60-month loans default more often at every verification level")
        save(fig, "06_term_verification.png", outdir)

    # 7 — class balance
    fig, ax = plt.subplots(figsize=(4.6, 4))
    counts = df[target].value_counts().sort_index()
    ax.bar(["Repaid", "Defaulted"], counts.values, color=[LOW, HIGH])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v/counts.sum():.1%})", ha="center", va="bottom",
                fontsize=9)
    ax.set_ylabel("Loans")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Class imbalance")
    save(fig, "07_class_balance.png", outdir)

    print(f"\n  {len(list(outdir.glob('*.png')))} figures in {outdir}")


if __name__ == "__main__":
    main()
