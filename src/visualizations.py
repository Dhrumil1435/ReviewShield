"""
visualizations.py
Visualization generation module for ReviewShield.
Provides helper functions for Word Cloud keyword generation and Stylometric Radar Charts.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Dict, Any, Tuple


def generate_wordclouds_from_weights(word_coef_map: Dict[str, float]) -> Tuple[plt.Figure, plt.Figure]:
    """
    Generates two WordCloud figures:
    1. Deceptive Keywords WordCloud (words with positive coefficients > 0)
    2. Authentic Keywords WordCloud (words with negative coefficients < 0)
    """
    deceptive_freqs = {word: weight for word, weight in word_coef_map.items() if weight > 0.05}
    authentic_freqs = {word: abs(weight) for word, weight in word_coef_map.items() if weight < -0.05}

    if not deceptive_freqs:
        deceptive_freqs = {"must_buy": 1.0, "flawless": 0.8, "life_changing": 0.7, "incredible": 0.6}
    if not authentic_freqs:
        authentic_freqs = {"decent": 1.0, "average": 0.8, "battery": 0.7, "price": 0.6}

    # Deceptive WordCloud (Red Palette)
    wc_deceptive = WordCloud(
        width=600,
        height=350,
        background_color="#1C1C1C",
        colormap="Reds",
        max_words=100,
        prefer_horizontal=0.8,
    ).generate_from_frequencies(deceptive_freqs)

    fig_dec, ax_dec = plt.subplots(figsize=(6, 3.5), facecolor="#1C1C1C")
    ax_dec.imshow(wc_deceptive, interpolation="bilinear")
    ax_dec.axis("off")
    ax_dec.set_title("Deceptive Trigger Keywords", color="#FCA5A5", fontsize=14, pad=12, fontweight="bold")
    fig_dec.tight_layout(pad=0)

    # Authentic WordCloud (Green Palette)
    wc_authentic = WordCloud(
        width=600,
        height=350,
        background_color="#1C1C1C",
        colormap="Greens",
        max_words=100,
        prefer_horizontal=0.8,
    ).generate_from_frequencies(authentic_freqs)

    fig_auth, ax_auth = plt.subplots(figsize=(6, 3.5), facecolor="#1C1C1C")
    ax_auth.imshow(wc_authentic, interpolation="bilinear")
    ax_auth.axis("off")
    ax_auth.set_title("Authentic Signal Keywords", color="#6EE7B7", fontsize=14, pad=12, fontweight="bold")
    fig_auth.tight_layout(pad=0)

    return fig_dec, fig_auth


def generate_stylometric_radar_chart(features: Dict[str, float]) -> plt.Figure:
    """
    Generates a polar Radar / Spider Chart comparing the review's stylometric feature vector
    against standard baseline thresholds.
    """
    categories = [
        "Vocab Diversity",
        "Readability (Norm)",
        "VADER Sentiment",
        "RoBERTa Sentiment",
        "Sentiment Gap",
        "Dissonance",
    ]
    N = len(categories)

    # Normalize values between 0.0 and 1.0 for polar plot
    vocab = min(max(features.get("vocab_diversity", 0.5), 0.0), 1.0)
    readability = min(max(features.get("readability_score", 50.0) / 100.0, 0.0), 1.0)
    vader = min(max((features.get("sentiment_score", 0.0) + 1.0) / 2.0, 0.0), 1.0)
    roberta = min(max((features.get("roberta_sentiment", 0.0) + 1.0) / 2.0, 0.0), 1.0)
    gap = min(max(features.get("roberta_rating_sentiment_gap", 0.0) / 2.0, 0.0), 1.0)
    dissonance = min(max(features.get("vader_roberta_dissonance", 0.0) / 2.0, 0.0), 1.0)

    values = [vocab, readability, vader, roberta, gap, dissonance]
    values += values[:1]  # Complete loop

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), subplot_kw=dict(polar=True), facecolor="#FFFFFF")
    ax.set_facecolor("#F9FAFB")

    # Draw polygon outlines
    ax.plot(angles, values, linewidth=2.5, linestyle="solid", color="#10B981", label="Current Review Profile")
    ax.fill(angles, values, color="#10B981", alpha=0.25)

    # Baseline Threshold (Reference Loop)
    baseline = [0.5, 0.6, 0.5, 0.5, 0.2, 0.15]
    baseline += baseline[:1]
    ax.plot(angles, baseline, linewidth=1.5, linestyle="dashed", color="#9CA3AF", label="Authentic Baseline")

    # Set Category Labels
    plt.xticks(angles[:-1], categories, color="#374151", size=9, fontweight="bold")
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="#9CA3AF", size=7)
    plt.ylim(0, 1.0)

    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    plt.title("Stylometric & Sentiment Radar Profile", fontsize=11, fontweight="bold", pad=16, color="#111827")
    fig.tight_layout()
    fig.tight_layout()

    return fig
