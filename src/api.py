"""FastAPI application for model explainability.

Exposes REST endpoints for SHAP explanations, LIME explanations,
prediction reports, and model-level reports.
"""

import io
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.shap_explainer import SHAPExplainer
from src.lime_explainer import LIMEExplainer
from src.report_generator import ExplainabilityReportGenerator

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as fh:
            return yaml.safe_load(fh) or {}
    return {}


config = _load_config()

app = FastAPI(
    title="Model Explainability API",
    description="SHAP and LIME explanations for ML model predictions.",
    version="1.0.0",
)

_state: Dict[str, Any] = {
    "model": None,
    "shap_explainer": None,
    "lime_explainer": None,
    "report_generator": None,
    "feature_names": [],
}


class FeaturesRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature name-value pairs")
    num_features: int = Field(default=10, description="Number of features in explanation")


class BatchFeaturesRequest(BaseModel):
    instances: List[Dict[str, float]] = Field(..., description="List of feature dicts")
    max_samples: int = Field(default=100, description="Maximum instances to explain")


class ModelLoadRequest(BaseModel):
    model_path: str = Field(..., description="Path to pickled model file")
    train_data_path: str = Field(..., description="Path to training data Parquet file")
    test_data_path: Optional[str] = Field(None, description="Path to test data Parquet file")
    target_column: str = Field(default="target", description="Target column name")


class ReportRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature name-value pairs")
    output_path: Optional[str] = Field(None, description="File path for saved report")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]
    feature_count: int


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health and model status."""
    model = _state["model"]
    return HealthResponse(
        status="healthy" if model is not None else "no_model_loaded",
        model_loaded=model is not None,
        model_type=type(model).__name__ if model else None,
        feature_count=len(_state["feature_names"]),
    )


@app.post("/load-model")
async def load_model(request: ModelLoadRequest) -> Dict[str, str]:
    """Load a model and initialize explainers.

    Expects a pickled scikit-learn compatible model and a Parquet file
    with training data.
    """
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
    if not os.path.exists(request.train_data_path):
        raise HTTPException(status_code=404, detail=f"Training data not found: {request.train_data_path}")

    with open(request.model_path, "rb") as fh:
        model = pickle.load(fh)

    train_df = pd.read_parquet(request.train_data_path)
    target_col = request.target_column

    if target_col in train_df.columns:
        X_train = train_df.drop(columns=[target_col])
    else:
        X_train = train_df

    feature_names = list(X_train.select_dtypes(include=[np.number]).columns)
    X_train_numeric = X_train[feature_names].fillna(0)

    shap_exp = SHAPExplainer(model=model, X_train=X_train_numeric, config_path=CONFIG_PATH, feature_names=feature_names)
    shap_exp.setup_explainer()

    lime_exp = LIMEExplainer(model=model, X_train=X_train_numeric, config_path=CONFIG_PATH, feature_names=feature_names)
    lime_exp.setup_explainer()

    _state["model"] = model
    _state["shap_explainer"] = shap_exp
    _state["lime_explainer"] = lime_exp
    _state["feature_names"] = feature_names

    if request.test_data_path and os.path.exists(request.test_data_path):
        test_df = pd.read_parquet(request.test_data_path)
        if target_col in test_df.columns:
            X_test = test_df[feature_names].fillna(0)
            y_test = test_df[target_col]
            _state["report_generator"] = ExplainabilityReportGenerator(
                model=model,
                X_train=X_train_numeric,
                X_test=X_test,
                y_test=y_test,
                config_path=CONFIG_PATH,
            )

    return {"status": "model_loaded", "features": str(feature_names), "model_type": type(model).__name__}


def _features_to_df(features: Dict[str, float]) -> pd.DataFrame:
    """Convert a feature dict to a single-row DataFrame."""
    feature_names = _state["feature_names"]
    values = {name: features.get(name, 0.0) for name in feature_names}
    return pd.DataFrame([values])


@app.post("/explain/shap")
async def shap_explain(request: FeaturesRequest) -> Dict[str, Any]:
    """Generate SHAP explanation for a single prediction."""
    if _state["shap_explainer"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST to /load-model first.")

    df = _features_to_df(request.features)
    explanation = _state["shap_explainer"].explain_prediction(df.iloc[0])

    return {
        "base_value": explanation["base_value"],
        "prediction": explanation["prediction"],
        "feature_contributions": explanation["feature_contributions"][:request.num_features],
        "text_explanation": explanation["text_explanation"],
    }


@app.post("/explain/lime")
async def lime_explain(request: FeaturesRequest) -> Dict[str, Any]:
    """Generate LIME explanation for a single prediction."""
    if _state["lime_explainer"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST to /load-model first.")

    df = _features_to_df(request.features)
    explanation = _state["lime_explainer"].explain_prediction(df.iloc[0], num_features=request.num_features)

    result = {
        "prediction": explanation["prediction"],
        "local_prediction": explanation["local_prediction"],
        "intercept": explanation["intercept"],
        "r_squared": explanation["r_squared"],
        "feature_weights": explanation["feature_weights"],
    }
    return result


@app.post("/report/prediction")
async def prediction_report(request: ReportRequest) -> Dict[str, Any]:
    """Generate a per-prediction explanation report (HTML)."""
    if _state["shap_explainer"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _features_to_df(request.features)
    explanation = _state["shap_explainer"].explain_prediction(df.iloc[0])

    generator = _state.get("report_generator")
    if generator is None:
        raise HTTPException(status_code=503, detail="Report generator not initialized (test data required)")

    html = generator.generate_prediction_report(
        instance=df.iloc[0],
        explanation=explanation,
        output_path=request.output_path,
    )

    return {"html": html, "output_path": request.output_path}


@app.post("/report/model")
async def model_report() -> Dict[str, Any]:
    """Generate a model-level report with global feature importance."""
    if _state["shap_explainer"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    generator = _state.get("report_generator")
    if generator is None:
        raise HTTPException(status_code=503, detail="Report generator not initialized")

    importance = _state["shap_explainer"].global_feature_importance(
        generator.X_test, max_samples=200
    )

    output_path = os.path.join(config.get("reporting", {}).get("output_dir", "./reports"), "model_report.html")
    html = generator.generate_global_report(importance_df=importance, output_path=output_path)

    return {"output_path": output_path, "feature_importance": importance.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
