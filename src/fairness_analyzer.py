"""Fairness and bias analysis for ML models.

Provides the FairnessAnalyzer class for computing disparate impact,
equalized odds, demographic parity, calibration across subgroups,
and generating multi-panel fairness dashboards.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class FairnessAnalyzer:
    """Evaluates model fairness across protected attributes.

    Computes standard fairness metrics including disparate impact,
    equalized odds, demographic parity, and calibration parity. Produces
    summary dashboards for compliance and audit documentation.

    Attributes:
        y_true: Ground truth labels.
        y_pred: Model predictions.
        y_proba: Predicted probabilities (optional).
        protected_attributes: DataFrame of protected attribute columns.
        favorable_outcome: The positive/favorable label value.
        threshold: Minimum acceptable fairness ratio.
    """

    def __init__(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        protected_attributes: pd.DataFrame,
        y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
        favorable_outcome: int = 1,
        threshold: float = 0.8,
    ) -> None:
        """Initialize the fairness analyzer.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            protected_attributes: DataFrame with one column per protected attribute.
            y_proba: Predicted probabilities for the positive class.
            favorable_outcome: The label value considered favorable.
            threshold: Minimum acceptable disparate impact ratio.
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba) if y_proba is not None else None
        self.protected_attributes = protected_attributes.copy()
        self.favorable_outcome = favorable_outcome
        self.threshold = threshold

        logger.info(
            "FairnessAnalyzer initialized: %d samples, %d protected attributes, threshold=%.2f",
            len(self.y_true),
            len(self.protected_attributes.columns),
            self.threshold,
        )

    def compute_disparate_impact(self, attribute: str) -> Dict[str, Any]:
        """Compute disparate impact ratio for a protected attribute.

        Disparate impact is the ratio of the positive outcome rate for the
        disadvantaged group to the advantaged group. A ratio below the
        threshold (commonly 0.8) indicates potential adverse impact.

        Args:
            attribute: Column name of the protected attribute.

        Returns:
            Dict with group_rates, ratio, passes_threshold, and
            reference_group.
        """
        groups = self.protected_attributes[attribute]
        unique_groups = groups.unique()

        group_rates = {}
        for group in sorted(unique_groups, key=str):
            mask = groups == group
            positive_rate = float((self.y_true[mask] == self.favorable_outcome).mean())
            group_rates[str(group)] = positive_rate

        sorted_groups = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
        reference_group = sorted_groups[0][0]
        reference_rate = sorted_groups[0][1]

        if reference_rate == 0:
            ratio = 1.0
        else:
            min_rate = min(group_rates.values())
            ratio = min_rate / reference_rate

        return {
            "group_rates": group_rates,
            "ratio": round(float(ratio), 4),
            "passes_threshold": ratio >= self.threshold,
            "reference_group": reference_group,
            "threshold": self.threshold,
        }

    def compute_equalized_odds(self, attribute: str) -> Dict[str, Any]:
        """Compute equalized odds metrics for a protected attribute.

        Equalized odds requires that the true positive rate (TPR) and
        false positive rate (FPR) are approximately equal across groups.

        Args:
            attribute: Column name of the protected attribute.

        Returns:
            Dict with per-group TPR, FPR, and gap metrics.
        """
        groups = self.protected_attributes[attribute]
        unique_groups = groups.unique()

        group_metrics = {}
        for group in sorted(unique_groups, key=str):
            mask = groups == group
            y_t = self.y_true[mask]
            y_p = self.y_pred[mask]

            tp = int(((y_p == self.favorable_outcome) & (y_t == self.favorable_outcome)).sum())
            fn = int(((y_p != self.favorable_outcome) & (y_t == self.favorable_outcome)).sum())
            fp = int(((y_p == self.favorable_outcome) & (y_t != self.favorable_outcome)).sum())
            tn = int(((y_p != self.favorable_outcome) & (y_t != self.favorable_outcome)).sum())

            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)

            group_metrics[str(group)] = {
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }

        tpr_values = [m["tpr"] for m in group_metrics.values()]
        fpr_values = [m["fpr"] for m in group_metrics.values()]

        tpr_gap = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
        fpr_gap = max(fpr_values) - min(fpr_values) if fpr_values else 0.0

        return {
            "group_metrics": group_metrics,
            "tpr_gap": round(float(tpr_gap), 4),
            "fpr_gap": round(float(fpr_gap), 4),
            "equalized": tpr_gap < 0.1 and fpr_gap < 0.1,
        }

    def compute_demographic_parity(self, attribute: str) -> Dict[str, Any]:
        """Compute demographic parity for a protected attribute.

        Demographic parity requires that the positive prediction rate
        is approximately equal across all groups, regardless of the
        true label.

        Args:
            attribute: Column name of the protected attribute.

        Returns:
            Dict with per-group positive prediction rates and max difference.
        """
        groups = self.protected_attributes[attribute]
        unique_groups = groups.unique()

        group_rates = {}
        for group in sorted(unique_groups, key=str):
            mask = groups == group
            pos_rate = float((self.y_pred[mask] == self.favorable_outcome).mean())
            group_rates[str(group)] = round(pos_rate, 4)

        rates = list(group_rates.values())
        max_diff = max(rates) - min(rates) if rates else 0.0

        return {
            "group_rates": group_rates,
            "max_difference": round(float(max_diff), 4),
            "parity_achieved": max_diff < 0.1,
        }

    def compute_calibration(self, attribute: str, n_bins: int = 10) -> Dict[str, Any]:
        """Compute calibration across subgroups.

        Measures how well predicted probabilities align with observed
        frequencies within each protected group.

        Args:
            attribute: Column name of the protected attribute.
            n_bins: Number of probability bins for calibration analysis.

        Returns:
            Dict with per-group calibration error metrics.
        """
        if self.y_proba is None:
            logger.warning("Predicted probabilities required for calibration analysis")
            return {"error": "no_probabilities", "group_calibration": {}}

        groups = self.protected_attributes[attribute]
        unique_groups = groups.unique()

        group_calibration = {}
        for group in sorted(unique_groups, key=str):
            mask = groups == group
            proba = self.y_proba[mask]
            true = self.y_true[mask]

            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(proba, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            bin_means_pred = []
            bin_means_true = []
            for b in range(n_bins):
                bin_mask = bin_indices == b
                if bin_mask.sum() > 0:
                    bin_means_pred.append(float(proba[bin_mask].mean()))
                    bin_means_true.append(float((true[bin_mask] == self.favorable_outcome).mean()))

            if bin_means_pred:
                ece = float(np.mean(np.abs(np.array(bin_means_pred) - np.array(bin_means_true))))
            else:
                ece = 0.0

            group_calibration[str(group)] = {
                "expected_calibration_error": round(ece, 4),
                "n_samples": int(mask.sum()),
                "mean_predicted_probability": round(float(proba.mean()), 4),
                "observed_positive_rate": round(float((true == self.favorable_outcome).mean()), 4),
            }

        ece_values = [g["expected_calibration_error"] for g in group_calibration.values()]
        max_ece_gap = max(ece_values) - min(ece_values) if ece_values else 0.0

        return {
            "group_calibration": group_calibration,
            "max_ece_gap": round(float(max_ece_gap), 4),
            "well_calibrated": max_ece_gap < 0.05,
        }

    def generate_fairness_metrics(self) -> Dict[str, Any]:
        """Compute all fairness metrics for every protected attribute.

        Returns:
            Nested dict keyed by attribute name, each containing
            disparate_impact, equalized_odds, demographic_parity,
            and calibration results.
        """
        results = {}

        for attr in self.protected_attributes.columns:
            logger.info("Computing fairness metrics for '%s'", attr)
            results[attr] = {
                "disparate_impact": self.compute_disparate_impact(attr),
                "equalized_odds": self.compute_equalized_odds(attr),
                "demographic_parity": self.compute_demographic_parity(attr),
                "calibration": self.compute_calibration(attr),
            }

        n_attrs = len(results)
        passing = sum(
            1 for attr_results in results.values()
            if attr_results["disparate_impact"]["passes_threshold"]
        )
        results["_summary"] = {
            "total_attributes": n_attrs,
            "passing_disparate_impact": passing,
            "overall_fair": passing == n_attrs,
        }

        return results

    def plot_fairness_dashboard(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
    ) -> None:
        """Generate a multi-panel fairness visualization dashboard.

        Creates a figure with panels for disparate impact, demographic
        parity, equalized odds (TPR/FPR), and calibration for each
        protected attribute.

        Args:
            save_path: File path for the saved dashboard image.
            figsize: Figure size in inches.
        """
        attributes = list(self.protected_attributes.columns)
        n_attrs = len(attributes)

        fig, axes = plt.subplots(n_attrs, 4, figsize=figsize, squeeze=False)
        fig.suptitle("Fairness Dashboard", fontsize=16, fontweight="bold", y=1.02)

        column_titles = ["Disparate Impact", "Demographic Parity", "Equalized Odds (TPR/FPR)", "Calibration"]
        for j, title in enumerate(column_titles):
            axes[0, j].set_title(title, fontweight="bold", fontsize=11)

        for i, attr in enumerate(attributes):
            di = self.compute_disparate_impact(attr)
            dp = self.compute_demographic_parity(attr)
            eo = self.compute_equalized_odds(attr)
            cal = self.compute_calibration(attr)

            ax = axes[i, 0]
            groups = list(di["group_rates"].keys())
            rates = list(di["group_rates"].values())
            colors = ["#2ecc71" if r >= self.threshold * max(rates) else "#e74c3c" for r in rates]
            ax.barh(groups, rates, color=colors)
            ax.set_xlabel("Positive Outcome Rate")
            ax.set_ylabel(attr)
            ax.axvline(x=self.threshold * max(rates) if rates else 0, color="red", linestyle="--", alpha=0.7)

            ax = axes[i, 1]
            dp_groups = list(dp["group_rates"].keys())
            dp_rates = list(dp["group_rates"].values())
            ax.barh(dp_groups, dp_rates, color="#3498db")
            ax.set_xlabel("Positive Prediction Rate")

            ax = axes[i, 2]
            eo_groups = list(eo["group_metrics"].keys())
            tpr_vals = [eo["group_metrics"][g]["tpr"] for g in eo_groups]
            fpr_vals = [eo["group_metrics"][g]["fpr"] for g in eo_groups]
            x_pos = np.arange(len(eo_groups))
            bar_width = 0.35
            ax.bar(x_pos - bar_width / 2, tpr_vals, bar_width, label="TPR", color="#27ae60")
            ax.bar(x_pos + bar_width / 2, fpr_vals, bar_width, label="FPR", color="#e74c3c")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(eo_groups, rotation=45, ha="right")
            ax.set_ylabel("Rate")
            ax.legend(fontsize=8)

            ax = axes[i, 3]
            if "group_calibration" in cal and cal.get("error") is None:
                cal_groups = list(cal["group_calibration"].keys())
                ece_vals = [cal["group_calibration"][g]["expected_calibration_error"] for g in cal_groups]
                colors = ["#f39c12" if v > 0.05 else "#27ae60" for v in ece_vals]
                ax.barh(cal_groups, ece_vals, color=colors)
                ax.set_xlabel("ECE")
            else:
                ax.text(0.5, 0.5, "No probabilities\navailable", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="#7f8c8d")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Fairness dashboard saved to %s", save_path)

        plt.close(fig)
