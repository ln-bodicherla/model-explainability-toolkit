"""SHAP-based model explainability.

Provides the SHAPExplainer class for computing SHAP values, generating
feature importance rankings, interaction analysis, visualizations, and
natural language explanations suitable for auditors and regulators.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

logger = logging.getLogger(__name__)


def _detect_model_type(model: Any) -> str:
    """Determine the model type for SHAP explainer selection."""
    model_class = type(model).__name__.lower()
    module = type(model).__module__ or ""

    tree_keywords = [
        "forest", "tree", "gbm", "gradient", "xgb", "lgbm",
        "lightgbm", "catboost", "booster",
    ]
    for keyword in tree_keywords:
        if keyword in model_class or keyword in module:
            return "tree"

    deep_keywords = ["neural", "sequential", "module", "keras", "torch"]
    for keyword in deep_keywords:
        if keyword in model_class or keyword in module:
            return "deep"

    return "kernel"


class SHAPExplainer:
    """SHAP-based model explanation engine.

    Automatically selects the appropriate SHAP explainer based on the model
    type (TreeExplainer for tree-based models, KernelExplainer for
    model-agnostic use, DeepExplainer for neural networks).

    Attributes:
        model: The trained model to explain.
        feature_names: List of feature names.
        explainer: The SHAP explainer instance.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        config_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the SHAP explainer.

        Args:
            model: Trained model with predict/predict_proba methods.
            X_train: Training data used as background for SHAP computation.
            config_path: Path to YAML configuration file.
            feature_names: Feature name list. Inferred from X_train if None.
        """
        self.model = model
        self.config = self._load_config(config_path)
        shap_cfg = self.config.get("shap", {})
        self.max_samples = shap_cfg.get("max_samples", 1000)
        self.check_additivity = shap_cfg.get("check_additivity", False)

        if len(X_train) > self.max_samples:
            self.background_data = X_train.sample(n=self.max_samples, random_state=42)
        else:
            self.background_data = X_train.copy()

        self.feature_names = feature_names or list(X_train.columns)
        self.model_type = _detect_model_type(model)
        self.explainer = None

        logger.info("SHAPExplainer initialized: model_type=%s features=%d background=%d",
                     self.model_type, len(self.feature_names), len(self.background_data))

    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> dict:
        if config_path is None:
            default = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            config_path = str(default)
        if os.path.exists(config_path):
            with open(config_path, "r") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    def setup_explainer(self) -> None:
        """Create the appropriate SHAP explainer based on model type.

        Tries TreeExplainer first for tree-based models, falls back to
        KernelExplainer if it fails.
        """
        if self.model_type == "tree":
            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer")
                return
            except Exception as exc:
                logger.warning("TreeExplainer failed (%s), falling back to KernelExplainer", exc)

        if self.model_type == "deep":
            try:
                self.explainer = shap.DeepExplainer(self.model, self.background_data.values)
                logger.info("Using DeepExplainer")
                return
            except Exception as exc:
                logger.warning("DeepExplainer failed (%s), falling back to KernelExplainer", exc)

        predict_fn = getattr(self.model, "predict_proba", self.model.predict)
        background = shap.sample(self.background_data, min(100, len(self.background_data)))
        self.explainer = shap.KernelExplainer(predict_fn, background)
        logger.info("Using KernelExplainer")

    def _ensure_setup(self) -> None:
        if self.explainer is None:
            raise RuntimeError("Explainer not set up. Call setup_explainer() first.")

    def _to_dataframe(self, instance: Union[pd.Series, pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert an instance to a single-row DataFrame."""
        if isinstance(instance, pd.Series):
            return instance.to_frame().T
        if isinstance(instance, np.ndarray):
            return pd.DataFrame([instance], columns=self.feature_names)
        return instance

    def explain_prediction(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
    ) -> Dict[str, Any]:
        """Compute SHAP values for a single prediction.

        Args:
            instance: Feature values for the prediction to explain.

        Returns:
            Dict with shap_values (per-feature contributions),
            base_value (expected output), prediction, and
            feature_contributions (sorted list of dicts).
        """
        self._ensure_setup()
        df = self._to_dataframe(instance)

        shap_values = self.explainer.shap_values(df, check_additivity=self.check_additivity)

        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif shap_values.ndim > 1:
            sv = shap_values[0]
        else:
            sv = shap_values

        if hasattr(self.explainer, "expected_value"):
            ev = self.explainer.expected_value
            base_value = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else float(np.array(ev).flat[0])
        else:
            base_value = 0.0

        contributions = []
        for i, name in enumerate(self.feature_names):
            contributions.append({
                "feature": name,
                "value": float(df.iloc[0, i]) if i < len(df.columns) else None,
                "shap_value": float(sv[i]),
                "abs_shap": abs(float(sv[i])),
            })
        contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

        prediction = None
        try:
            pred = self.model.predict(df)
            prediction = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        except Exception:
            pass

        text_explanation = self.generate_explanation_text(contributions, base_value, prediction)

        return {
            "shap_values": sv.tolist() if isinstance(sv, np.ndarray) else list(sv),
            "base_value": base_value,
            "prediction": prediction,
            "feature_contributions": contributions,
            "text_explanation": text_explanation,
        }

    def explain_batch(
        self,
        X: pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compute SHAP explanations for a batch of instances.

        Args:
            X: DataFrame of instances to explain.
            max_samples: Maximum number of samples to explain.

        Returns:
            List of explanation dicts (same format as explain_prediction).
        """
        if max_samples and len(X) > max_samples:
            X = X.head(max_samples)

        results = []
        for idx in range(len(X)):
            explanation = self.explain_prediction(X.iloc[idx])
            explanation["index"] = idx
            results.append(explanation)
            if (idx + 1) % 50 == 0:
                logger.info("Explained %d/%d instances", idx + 1, len(X))

        return results

    def global_feature_importance(
        self,
        X: pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute global feature importance as mean absolute SHAP values.

        Args:
            X: Dataset to compute importance over.
            max_samples: Subsample size.

        Returns:
            DataFrame with feature, mean_abs_shap, and rank columns,
            sorted by importance.
        """
        self._ensure_setup()

        if max_samples and len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)

        shap_values = self.explainer.shap_values(X, check_additivity=self.check_additivity)

        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["cumulative_pct"] = (
            importance_df["mean_abs_shap"].cumsum() / importance_df["mean_abs_shap"].sum() * 100
        ).round(2)

        return importance_df

    def feature_interaction(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> pd.DataFrame:
        """Compute SHAP interaction values for feature pairs.

        Only supported for TreeExplainer models. Returns a DataFrame of
        the top interaction pairs by mean absolute interaction value.

        Args:
            X: Dataset to compute interactions over.
            max_samples: Subsample size.

        Returns:
            DataFrame with feature_1, feature_2, and interaction_value.
        """
        self._ensure_setup()

        if not isinstance(self.explainer, shap.TreeExplainer):
            logger.warning("Interaction values only supported for TreeExplainer")
            return pd.DataFrame(columns=["feature_1", "feature_2", "interaction_value"])

        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)

        interaction_values = self.explainer.shap_interaction_values(X)

        if isinstance(interaction_values, list):
            iv = interaction_values[1] if len(interaction_values) > 1 else interaction_values[0]
        else:
            iv = interaction_values

        mean_interactions = np.abs(iv).mean(axis=0)
        n_features = len(self.feature_names)

        pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                pairs.append({
                    "feature_1": self.feature_names[i],
                    "feature_2": self.feature_names[j],
                    "interaction_value": float(mean_interactions[i, j]),
                })

        interactions_df = pd.DataFrame(pairs).sort_values(
            "interaction_value", ascending=False
        ).reset_index(drop=True)

        return interactions_df

    def plot_waterfall(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
        save_path: Optional[str] = None,
        max_display: int = 15,
    ) -> None:
        """Generate a SHAP waterfall plot for a single prediction."""
        self._ensure_setup()
        df = self._to_dataframe(instance)

        shap_values = self.explainer.shap_values(df, check_additivity=self.check_additivity)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0] if shap_values.ndim > 1 else shap_values

        ev = self.explainer.expected_value
        base_val = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else float(np.array(ev).flat[0])

        explanation = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=df.values[0],
            feature_names=self.feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Waterfall plot saved to %s", save_path)
        plt.close(fig)

    def plot_force(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
        save_path: Optional[str] = None,
    ) -> None:
        """Generate a SHAP force plot for a single prediction."""
        self._ensure_setup()
        df = self._to_dataframe(instance)

        shap_values = self.explainer.shap_values(df, check_additivity=self.check_additivity)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0] if shap_values.ndim > 1 else shap_values

        ev = self.explainer.expected_value
        base_val = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else float(np.array(ev).flat[0])

        force = shap.force_plot(base_val, sv, df.iloc[0], feature_names=self.feature_names, matplotlib=True, show=False)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Force plot saved to %s", save_path)
        plt.close("all")

    def plot_beeswarm(
        self,
        X: pd.DataFrame,
        save_path: Optional[str] = None,
        max_display: int = 20,
    ) -> None:
        """Generate a SHAP beeswarm plot for global feature importance."""
        self._ensure_setup()

        shap_values = self.explainer.shap_values(X, check_additivity=self.check_additivity)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        explanation = shap.Explanation(
            values=sv,
            base_values=np.full(len(sv), float(np.array(self.explainer.expected_value).flat[0])),
            data=X.values,
            feature_names=self.feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Beeswarm plot saved to %s", save_path)
        plt.close(fig)

    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Generate a SHAP dependence plot for a feature."""
        self._ensure_setup()

        shap_values = self.explainer.shap_values(X, check_additivity=self.check_additivity)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            sv,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            ax=ax,
            show=False,
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Dependence plot saved to %s", save_path)
        plt.close(fig)

    def generate_explanation_text(
        self,
        contributions: List[Dict],
        base_value: float,
        prediction: Optional[float] = None,
        top_n: int = 5,
    ) -> str:
        """Generate a natural language explanation of a prediction.

        Produces a paragraph suitable for auditors and non-technical
        stakeholders describing which features drove the prediction
        and in which direction.

        Args:
            contributions: Sorted list of feature contribution dicts.
            base_value: Model's expected (baseline) output.
            prediction: The actual prediction value.
            top_n: Number of top features to mention.

        Returns:
            Human-readable explanation string.
        """
        lines = []
        lines.append(f"The model's baseline prediction is {base_value:.4f}.")

        if prediction is not None:
            lines.append(f"For this instance, the model predicted {prediction:.4f}.")

        top = contributions[:top_n]
        positive = [c for c in top if c["shap_value"] > 0]
        negative = [c for c in top if c["shap_value"] < 0]

        if positive:
            drivers = ", ".join(
                f"{c['feature']} (value={c['value']}, contribution=+{c['shap_value']:.4f})"
                for c in positive
            )
            lines.append(f"The following features pushed the prediction higher: {drivers}.")

        if negative:
            drivers = ", ".join(
                f"{c['feature']} (value={c['value']}, contribution={c['shap_value']:.4f})"
                for c in negative
            )
            lines.append(f"The following features pushed the prediction lower: {drivers}.")

        total_attribution = sum(c["shap_value"] for c in contributions)
        top_attribution = sum(c["shap_value"] for c in top)
        if total_attribution != 0:
            coverage = abs(top_attribution / total_attribution) * 100
            lines.append(f"The top {top_n} features account for {coverage:.1f}% of the total attribution.")

        return " ".join(lines)

    def export_to_json(
        self,
        explanation: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """Export an explanation to a structured JSON format.

        Args:
            explanation: Explanation dict from explain_prediction.
            output_path: Optional file path to write JSON.

        Returns:
            JSON string.
        """
        export = {
            "base_value": explanation["base_value"],
            "prediction": explanation["prediction"],
            "feature_contributions": explanation["feature_contributions"],
            "text_explanation": explanation["text_explanation"],
        }

        json_str = json.dumps(export, indent=2, default=str)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as fh:
                fh.write(json_str)
            logger.info("Explanation exported to %s", output_path)

        return json_str
