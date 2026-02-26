"""Uncertainty & calibration metrics + reliability diagram plotting.

Combines TransDiff-style metrics (AURC, EAURC, AUROC, AUPR, FPR@95, ECE, NLL,
Brier) with STELAR-style reliability diagram and bin-count distribution plots.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ===========================================================================
# TransDiff-style scalar metrics
# ===========================================================================


def calc_aurc_eaurc(
    softmax: np.ndarray,
    correct: np.ndarray,
) -> Tuple[float, float]:
    """Area Under Risk-Coverage curve and Excess AURC.

    Args:
        softmax: (N, K) softmax probabilities.
        correct: (N,) binary correctness array (1 = correct, 0 = wrong).

    Returns:
        (aurc, eaurc)
    """
    softmax_max = softmax.max(axis=1)
    n = len(correct)
    # Sort by descending confidence
    idx = np.argsort(-softmax_max)
    sorted_correct = correct[idx]

    # Cumulative risk at each coverage level
    risk_cov = np.cumsum(1 - sorted_correct) / np.arange(1, n + 1)
    aurc = risk_cov.mean()

    # Optimal AURC for error rate r
    r = 1.0 - correct.mean()
    if r > 0 and r < 1:
        optimal = r + (1 - r) * np.log(1 - r)
    else:
        optimal = 0.0
    eaurc = aurc - optimal
    return float(aurc), float(eaurc)


def calc_fpr_auroc_aupr(
    softmax: np.ndarray,
    correct: np.ndarray,
) -> Dict[str, float]:
    """FPR@95, AUROC, AUPR (success), AUPR (error).

    Uses max-softmax as the confidence score, correctness as binary label.

    Args:
        softmax: (N, K) softmax probabilities.
        correct: (N,) binary correctness (1 = correct).

    Returns:
        dict with keys: fpr95, auroc, aupr_success, aupr_error
    """
    softmax_max = softmax.max(axis=1)

    # AUROC
    fpr_arr, tpr_arr, _ = roc_curve(correct, softmax_max)
    auroc = auc(fpr_arr, tpr_arr)

    # FPR at 95% TPR
    fpr95 = fpr_arr[np.searchsorted(tpr_arr, 0.95)].item() if tpr_arr.max() >= 0.95 else 1.0

    # AUPR success (correct=1 as positive)
    prec, rec, _ = precision_recall_curve(correct, softmax_max)
    aupr_success = auc(rec, prec)

    # AUPR error (error=1 as positive, negate scores)
    aupr_error = average_precision_score(1 - correct, -softmax_max)

    return {
        "fpr95": float(fpr95),
        "auroc": float(auroc),
        "aupr_success": float(aupr_success),
        "aupr_error": float(aupr_error),
    }


def calc_ece(
    softmax: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected Calibration Error (equal-width bins).

    Args:
        softmax: (N, K) softmax probabilities.
        labels: (N,) ground-truth class indices.
        n_bins: number of bins.

    Returns:
        (ece, bin_acc, bin_conf, bin_count) — ece is scalar, arrays are (n_bins,).
    """
    softmax_max = softmax.max(axis=1)
    predictions = softmax.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        if b < n_bins - 1:
            mask = (softmax_max >= lo) & (softmax_max < hi)
        else:
            mask = (softmax_max >= lo) & (softmax_max <= hi)
        cnt = mask.sum()
        bin_count[b] = cnt
        if cnt > 0:
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = softmax_max[mask].mean()

    n = len(labels)
    ece = float(np.sum(bin_count / max(n, 1) * np.abs(bin_acc - bin_conf)))
    return ece, bin_acc, bin_conf, bin_count


def calc_nll_brier(
    softmax: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Negative log-likelihood and Brier score.

    Args:
        softmax: (N, K) softmax probabilities.
        logits: (N, K) raw logits.
        labels: (N,) ground-truth class indices.

    Returns:
        (nll, brier)
    """
    n, k = softmax.shape

    # NLL via log-softmax
    log_softmax = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-12)
    nll = -log_softmax[np.arange(n), labels].mean()

    # Brier score
    one_hot = np.eye(k)[labels]
    brier = np.mean(np.sum((softmax - one_hot) ** 2, axis=1))

    return float(nll), float(brier)


# ===========================================================================
# Compute all metrics at once
# ===========================================================================


def compute_all_metrics(
    logits_t: torch.Tensor,
    labels_t: torch.Tensor,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute all TransDiff-style metrics from logits and labels.

    Args:
        logits_t: (N, K) raw logits tensor.
        labels_t: (N,) ground-truth label tensor.
        n_bins: ECE bin count.

    Returns:
        dict with keys: accuracy, ece, nll, brier, aurc, eaurc, fpr95, auroc,
                         aupr_success, aupr_error
    """
    softmax_t = torch.softmax(logits_t.float(), dim=-1)
    softmax = softmax_t.cpu().numpy()
    logits = logits_t.cpu().float().numpy()
    labels = labels_t.cpu().numpy().astype(np.int64)

    predictions = softmax.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    accuracy = correct.mean() * 100.0

    ece, bin_acc, bin_conf, bin_count = calc_ece(softmax, labels, n_bins)
    nll, brier = calc_nll_brier(softmax, logits, labels)
    aurc, eaurc = calc_aurc_eaurc(softmax, correct)
    fpr_auroc_aupr = calc_fpr_auroc_aupr(softmax, correct)

    return {
        "accuracy": accuracy,
        "ece": ece,
        "nll": nll,
        "brier": brier,
        "aurc": aurc,
        "eaurc": eaurc,
        **fpr_auroc_aupr,
    }


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics dict into a human-readable table (TransDiff scaling)."""
    lines = [
        f"  Accuracy     : {metrics['accuracy']:.2f}%",
        f"  ECE          : {metrics['ece'] * 100:.2f}%",
        f"  NLL          : {metrics['nll']:.4f}",
        f"  Brier        : {metrics['brier'] * 100:.2f}%",
        f"  AURC         : {metrics['aurc'] * 1000:.2f}‰",
        f"  EAURC        : {metrics['eaurc'] * 1000:.2f}‰",
        f"  FPR@95       : {metrics['fpr95'] * 100:.2f}%",
        f"  AUROC        : {metrics['auroc'] * 100:.2f}%",
        f"  AUPR (Succ.) : {metrics['aupr_success'] * 100:.2f}%",
        f"  AUPR (Error) : {metrics['aupr_error'] * 100:.2f}%",
    ]
    return "\n".join(lines)


# ===========================================================================
# STELAR-style reliability diagram + bin count plot
# ===========================================================================


def plot_reliability_diagram(
    logits_t: torch.Tensor,
    labels_t: torch.Tensor,
    save_dir: str,
    split: str = "test",
    n_bins: int = 15,
) -> Dict[str, float]:
    """Plot reliability diagram and bin-count distribution, save PNG + CSV.

    Following the STELAR-private/REVE implementation:
      - Reliability diagram: blue accuracy bars + red gap overlay + diagonal
      - Bin count distribution: bar chart with count annotations
      - CSV with per-bin stats and summary metrics

    Args:
        logits_t: (N, K) raw logits tensor.
        labels_t: (N,) ground-truth label tensor.
        save_dir: directory to save plots and CSV.
        split: prefix for filenames (e.g., "test", "val").
        n_bins: number of equal-width bins.

    Returns:
        dict with ece, nll, brier values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calib_dir = Path(save_dir) / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Compute probabilities
    y_probs = torch.softmax(logits_t.float().cpu(), dim=-1)
    y_true = labels_t.cpu().long()
    N, K = y_probs.shape

    # NLL
    nll = F.cross_entropy(logits_t.float().cpu(), y_true).item()

    # Brier score
    y_onehot = F.one_hot(y_true, num_classes=K).float()
    brier = ((y_probs - y_onehot) ** 2).sum(dim=-1).mean().item()

    # ECE binning
    confidences, predictions = y_probs.max(dim=-1)
    accuracies = (predictions == y_true).float()

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = torch.zeros(n_bins)
    bin_conf = torch.zeros(n_bins)
    bin_count = torch.zeros(n_bins, dtype=torch.long)

    for b in range(n_bins):
        lo, hi = bin_boundaries[b].item(), bin_boundaries[b + 1].item()
        if b < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        cnt = mask.sum().item()
        bin_count[b] = int(cnt)
        if cnt > 0:
            bin_acc[b] = accuracies[mask].mean().item()
            bin_conf[b] = confidences[mask].mean().item()

    ece = (bin_count.float() / max(N, 1) * (bin_acc - bin_conf).abs()).sum().item()

    # --- Figure 1: Reliability Diagram ---
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    width = (1.0 / n_bins) * 0.8

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Blue bars: per-bin accuracy
    ax.bar(
        bin_centers.numpy(),
        bin_acc.numpy(),
        width=width,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        label="Accuracy",
        alpha=0.8,
    )
    # Diagonal: perfect calibration
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")

    # Red gap bars: |accuracy - confidence| overlay
    for b in range(n_bins):
        if bin_count[b] > 0:
            lo_v = min(bin_acc[b].item(), bin_conf[b].item())
            hi_v = max(bin_acc[b].item(), bin_conf[b].item())
            ax.bar(
                bin_centers[b].item(),
                hi_v - lo_v,
                bottom=lo_v,
                width=width,
                color="red",
                alpha=0.3,
                edgecolor="red",
                linewidth=0.5,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Reliability Diagram ({split})\nECE={ece:.4f}  NLL={nll:.4f}  Brier={brier:.4f}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(calib_dir / f"{split}_reliability_diagram.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: Bin Count Distribution ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    max_count = max(bin_count.max().item(), 1)
    colors = plt.cm.Blues(0.3 + 0.5 * bin_count.float().numpy() / max_count)
    ax2.bar(
        bin_centers.numpy(),
        bin_count.numpy(),
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    # Annotate counts
    for b in range(n_bins):
        cnt = int(bin_count[b].item())
        if cnt > 0:
            ax2.text(
                bin_centers[b].item(),
                cnt + max(max_count * 0.01, 0.5),
                str(cnt),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, max_count * 1.15)
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Sample Count", fontsize=12)
    ax2.set_title(f"Bin Count Distribution ({split})  —  N={N}", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(str(calib_dir / f"{split}_bin_count_distribution.png"), dpi=150)
    plt.close(fig2)

    # --- CSV ---
    csv_path = calib_dir / f"{split}_reliability_bins.csv"
    with open(csv_path, "w") as f:
        f.write("bin,bin_lower,bin_upper,count,avg_confidence,avg_accuracy,gap\n")
        for b in range(n_bins):
            lo = bin_boundaries[b].item()
            hi = bin_boundaries[b + 1].item()
            gap = abs(bin_acc[b].item() - bin_conf[b].item())
            f.write(
                f"{b},{lo:.4f},{hi:.4f},{int(bin_count[b].item())},"
                f"{bin_conf[b].item():.6f},{bin_acc[b].item():.6f},{gap:.6f}\n"
            )
        f.write(f"\n# ECE,{ece:.6f}\n")
        f.write(f"# NLL,{nll:.6f}\n")
        f.write(f"# Brier,{brier:.6f}\n")

    return {"ece": ece, "nll": nll, "brier": brier}


# ===========================================================================
# MC-stochastic evaluation utility
# ===========================================================================


def mc_forward_collect(
    model_fn,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_passes: int = 10,
) -> torch.Tensor:
    """Run `n_passes` stochastic forward passes and average logits.

    The model should have dropout/stochastic components active during eval
    (e.g., via `model.train()` or explicit dropout flags).

    Args:
        model_fn: callable(input_ids, attention_mask) -> logits (B, K)
        input_ids: (B, S)
        attention_mask: (B, S)
        n_passes: number of stochastic forward passes.

    Returns:
        (B, K) averaged logits.
    """
    all_logits = []
    for _ in range(n_passes):
        with torch.no_grad():
            logits = model_fn(input_ids, attention_mask)
        all_logits.append(logits)
    return torch.stack(all_logits, dim=0).mean(dim=0)
