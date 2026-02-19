"""LIME-based model explainability.

Provides the LIMEExplainer class for computing local interpretable
explanations using the LIME library. Supports tabular data, stability
analysis, and cross-framework comparison with SHAP.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """LIME-based local explanation engine for tabular models.

    Creates local surrogate models around individual predictions to
    explain which features contributed most to a particular outcome.

    Attributes:
        model: The trained model to explain.
        feature_names: List of feature column names.
        explainer: The LIME TabularExplainer instance.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        config_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "classification",
    ) -> None:
        """Initialize the LIME explainer.

        Args:
            model: Trained model with predict/predict_proba methods.
            X_train: Training data for computing statistics.
            config_path: Path to YAML configuration file.
            feature_names: Feature name list. Inferred from X_train if None.
            class_names: Class label names for classification tasks.
            mode: Either "classification" or "regression".
        """
        self.model = model
        self.X_train = X_train.copy()
        self.config = self._load_config(config_path)
        lime_cfg = self.config.get("lime", {})
        self.num_features = lime_cfg.get("num_features", 10)
        self.num_samples = lime_cfg.get("num_samples", 5000)

        self.feature_names = feature_names or list(X_train.columns)
        self.class_names = class_names or ["class_0", "class_1"]
        self.mode = mode
        self.explainer = None

        categorical_features = []
        for i, col in enumerate(self.feature_names):
            if X_train[col].dtype == "object" or X_train[col].nunique() < 10:
                categorical_features.append(i)
        self.categorical_features = categorical_features

        logger.info("LIMEExplainer initialized: mode=%s features=%d", mode, len(self.feature_names))

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
        """Create the LIME TabularExplainer."""
        self.explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=self.categorical_features if self.categorical_features else None,
            mode=self.mode,
            discretize_continuous=True,
            random_state=42,
        )
        logger.info("LIME TabularExplainer created")

    def _ensure_setup(self) -> None:
        if self.explainer is None:
            raise RuntimeError("Explainer not set up. Call setup_explainer() first.")

    def _get_predict_fn(self) -> Callable:
        """Return the appropriate prediction function for LIME."""
        if self.mode == "classification" and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        return self.model.predict

    def _to_array(self, instance: Union[pd.Series, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert instance to a 1-D numpy array."""
        if isinstance(instance, pd.DataFrame):
            return instance.iloc[0].values.astype(float)
        if isinstance(instance, pd.Series):
            return instance.values.astype(float)
        return np.array(instance, dtype=float).flatten()

    def explain_prediction(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None,
        labels: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        """Generate a LIME explanation for a single prediction.

        Args:
            instance: Feature values for the prediction to explain.
            num_features: Number of features in the explanation.
            num_samples: Number of perturbation samples.
            labels: Which class labels to explain (classification only).

        Returns:
            Dict with feature_weights, prediction, intercept, score,
            and the raw LIME explanation object.
        """
        self._ensure_setup()
        arr = self._to_array(instance)
        n_feat = num_features or self.num_features
        n_samp = num_samples or self.num_samples

        kwargs = {
            "data_row": arr,
            "predict_fn": self._get_predict_fn(),
            "num_features": n_feat,
            "num_samples": n_samp,
        }
        if labels and self.mode == "classification":
            kwargs["labels"] = labels

        explanation = self.explainer.explain_instance(**kwargs)

        if self.mode == "classification":
            label = labels[0] if labels else 1
            feature_weights = explanation.as_list(label=label)
            local_pred = explanation.local_pred[0] if hasattr(explanation, "local_pred") else None
            score = explanation.score if hasattr(explanation, "score") else None
            intercept = explanation.intercept.get(label) if hasattr(explanation, "intercept") else None
        else:
            feature_weights = explanation.as_list()
            local_pred = explanation.local_pred[0] if hasattr(explanation, "local_pred") else None
            score = explanation.score if hasattr(explanation, "score") else None
            intercept = explanation.intercept.get(1) if hasattr(explanation, "intercept") else None

        prediction = None
        try:
            pred_fn = self._get_predict_fn()
            pred = pred_fn(arr.reshape(1, -1))
            if pred.ndim > 1:
                prediction = float(pred[0, 1])
            else:
                prediction = float(pred[0])
        except Exception:
            pass

        return {
            "feature_weights": feature_weights,
            "prediction": prediction,
            "local_prediction": float(local_pred) if local_pred is not None else None,
            "intercept": float(intercept) if intercept is not None else None,
            "r_squared": float(score) if score is not None else None,
            "explanation_object": explanation,
        }

    def explain_batch(
        self,
        X: pd.DataFrame,
        num_features: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate LIME explanations for a batch of instances.

        Args:
            X: DataFrame of instances.
            num_features: Features per explanation.
            max_samples: Maximum instances to explain.

        Returns:
            List of explanation dicts.
        """
        if max_samples and len(X) > max_samples:
            X = X.head(max_samples)

        results = []
        for idx in range(len(X)):
            exp = self.explain_prediction(X.iloc[idx], num_features=num_features)
            del exp["explanation_object"]
            exp["index"] = idx
            results.append(exp)
            if (idx + 1) % 50 == 0:
                logger.info("LIME: explained %d/%d instances", idx + 1, len(X))

        return results

    def get_top_features(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get the top contributing features for a prediction.

        Args:
            instance: Feature values.
            n: Number of top features to return.

        Returns:
            Sorted list of dicts with feature, weight, and direction.
        """
        result = self.explain_prediction(instance, num_features=n)
        top = []
        for feat_desc, weight in result["feature_weights"][:n]:
            top.append({
                "feature": feat_desc,
                "weight": float(weight),
                "direction": "positive" if weight > 0 else "negative",
            })
        return top

    def plot_explanation(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
        save_path: Optional[str] = None,
        num_features: Optional[int] = None,
    ) -> None:
        """Generate and optionally save a LIME explanation bar chart.

        Args:
            instance: Feature values.
            save_path: File path for the saved plot.
            num_features: Number of features to display.
        """
        result = self.explain_prediction(instance, num_features=num_features)
        explanation = result["explanation_object"]

        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("LIME plot saved to %s", save_path)
        plt.close(fig)

    def compare_with_shap(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
        shap_explanation: Dict[str, Any],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Compare LIME and SHAP explanations side by side.

        Args:
            instance: Feature values.
            shap_explanation: Output of SHAPExplainer.explain_prediction.
            top_n: Number of top features to compare.

        Returns:
            Dict with per-feature comparison and agreement metrics.
        """
        lime_result = self.explain_prediction(instance, num_features=top_n)

        lime_features = {}
        for feat_desc, weight in lime_result["feature_weights"]:
            for name in self.feature_names:
                if name in feat_desc:
                    lime_features[name] = float(weight)
                    break

        shap_features = {}
        for contrib in shap_explanation.get("feature_contributions", [])[:top_n]:
            shap_features[contrib["feature"]] = contrib["shap_value"]

        all_features = set(lime_features.keys()) | set(shap_features.keys())
        comparison = []
        direction_agreements = 0
        total_compared = 0

        for feat in all_features:
            lime_val = lime_features.get(feat)
            shap_val = shap_features.get(feat)
            agree = None
            if lime_val is not None and shap_val is not None:
                agree = (lime_val > 0) == (shap_val > 0)
                direction_agreements += int(agree)
                total_compared += 1
            comparison.append({
                "feature": feat,
                "lime_weight": lime_val,
                "shap_value": shap_val,
                "direction_agree": agree,
            })

        comparison.sort(key=lambda x: abs(x.get("shap_value") or 0), reverse=True)

        lime_rank = [f for f, _ in lime_result["feature_weights"][:top_n]]
        shap_rank = [c["feature"] for c in shap_explanation.get("feature_contributions", [])[:top_n]]
        rank_overlap = len(set(lime_rank) & set(shap_rank))

        return {
            "comparison": comparison,
            "direction_agreement_rate": direction_agreements / max(total_compared, 1),
            "rank_overlap": rank_overlap,
            "rank_overlap_pct": rank_overlap / max(len(lime_rank), 1) * 100,
        }

    def stability_analysis(
        self,
        instance: Union[pd.Series, pd.DataFrame, np.ndarray],
        n_runs: int = 10,
        num_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Measure explanation stability across repeated runs.

        Runs LIME multiple times on the same instance with different
        random perturbations and measures how consistent the feature
        rankings and weights are.

        Args:
            instance: Feature values.
            n_runs: Number of repeated explanation runs.
            num_features: Features per explanation.

        Returns:
            Dict with per-feature weight statistics and overall
            stability score.
        """
        self._ensure_setup()
        n_feat = num_features or self.num_features

        all_weights: Dict[str, List[float]] = {}
        all_rankings: Dict[str, List[int]] = {}

        for run in range(n_runs):
            result = self.explain_prediction(
                instance, num_features=n_feat, num_samples=self.num_samples
            )
            for rank, (feat_desc, weight) in enumerate(result["feature_weights"]):
                matched_name = feat_desc
                for name in self.feature_names:
                    if name in feat_desc:
                        matched_name = name
                        break

                if matched_name not in all_weights:
                    all_weights[matched_name] = []
                    all_rankings[matched_name] = []
                all_weights[matched_name].append(float(weight))
                all_rankings[matched_name].append(rank)

        feature_stability = []
        for feat in all_weights:
            weights = np.array(all_weights[feat])
            rankings = np.array(all_rankings[feat])
            feature_stability.append({
                "feature": feat,
                "mean_weight": float(weights.mean()),
                "std_weight": float(weights.std()),
                "cv_weight": float(weights.std() / (abs(weights.mean()) + 1e-10)),
                "mean_rank": float(rankings.mean()),
                "std_rank": float(rankings.std()),
                "appearances": len(weights),
            })

        feature_stability.sort(key=lambda x: x["mean_rank"])

        cv_values = [f["cv_weight"] for f in feature_stability if f["appearances"] == n_runs]
        overall_stability = 1.0 - min(np.mean(cv_values), 1.0) if cv_values else 0.0

        return {
            "feature_stability": feature_stability,
            "overall_stability_score": float(overall_stability),
            "n_runs": n_runs,
            "interpretation": (
                "High stability" if overall_stability > 0.8
                else "Moderate stability" if overall_stability > 0.5
                else "Low stability -- consider increasing num_samples"
            ),
        }
