"""Explainability report generation.

Provides the ExplainabilityReportGenerator class for creating model cards,
per-prediction reports, global model behavior reports, fairness reports,
and regulatory compliance reports rendered with Jinja2 templates.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

DEFAULT_CSS = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; line-height: 1.6; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
h3 { color: #34495e; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th { background-color: #3498db; color: white; padding: 10px; text-align: left; }
td { border: 1px solid #ddd; padding: 8px; }
tr:nth-child(even) { background-color: #f9f9f9; }
.pass { color: #27ae60; font-weight: bold; }
.fail { color: #e74c3c; font-weight: bold; }
.warn { color: #f39c12; font-weight: bold; }
.metric-card { display: inline-block; background: #ecf0f1; padding: 15px 25px; margin: 5px; border-radius: 8px; text-align: center; }
.metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
.metric-label { font-size: 12px; color: #7f8c8d; }
.section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background: #f8f9fa; }
.footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #7f8c8d; }
"""


class ExplainabilityReportGenerator:
    """Generates explainability and compliance reports.

    Renders professional HTML reports using Jinja2 templates for model
    cards, prediction-level explanations, global behavior analysis,
    fairness assessments, and regulatory compliance documentation.

    Attributes:
        model: The trained model being documented.
        model_name: Human-readable model identifier.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config_path: Optional[str] = None,
        model_name: str = "Model",
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        self.config = self._load_config(config_path)
        reporting_cfg = self.config.get("reporting", {})
        self.template_dir = reporting_cfg.get("template_dir", "./templates")
        self.output_dir = reporting_cfg.get("output_dir", "./reports")
        self.include_plots = reporting_cfg.get("include_plots", True)
        self.output_format = reporting_cfg.get("format", "html")

        self.y_pred = model.predict(X_test)
        self.y_proba = None
        if hasattr(model, "predict_proba"):
            self.y_proba = model.predict_proba(X_test)

        self._performance_metrics = self._compute_performance()

    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> dict:
        if config_path is None:
            default = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            config_path = str(default)
        if os.path.exists(config_path):
            with open(config_path, "r") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    def _compute_performance(self) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(self.y_test, self.y_pred)),
            "precision": float(precision_score(self.y_test, self.y_pred, zero_division=0, average="weighted")),
            "recall": float(recall_score(self.y_test, self.y_pred, zero_division=0, average="weighted")),
            "f1": float(f1_score(self.y_test, self.y_pred, zero_division=0, average="weighted")),
        }
        if self.y_proba is not None:
            try:
                if self.y_proba.ndim > 1 and self.y_proba.shape[1] == 2:
                    metrics["auc_roc"] = float(roc_auc_score(self.y_test, self.y_proba[:, 1]))
                else:
                    metrics["auc_roc"] = float(roc_auc_score(self.y_test, self.y_proba, multi_class="ovr", average="weighted"))
            except ValueError:
                metrics["auc_roc"] = None
        return metrics

    def render_html(self, title: str, body_content: str, css: Optional[str] = None) -> str:
        """Wrap body content in a full HTML document.

        Args:
            title: Page title.
            body_content: Inner HTML body.
            css: Custom CSS (uses DEFAULT_CSS if None).

        Returns:
            Complete HTML document string.
        """
        style = css or DEFAULT_CSS
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{style}</style>
</head>
<body>
{body_content}
<div class="footer">
    <p>Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
</div>
</body>
</html>"""

    def _save_report(self, html: str, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as fh:
            fh.write(html)
        logger.info("Report saved to %s", output_path)
        return output_path

    def generate_model_card(self, output_path: Optional[str] = None) -> str:
        """Generate a model card documenting performance, features, and limitations.

        Args:
            output_path: File path for the HTML report.

        Returns:
            HTML content as a string.
        """
        model_type = type(self.model).__name__
        n_features = len(self.X_train.columns)
        feature_list = list(self.X_train.columns)

        metrics_html = ""
        for name, val in self._performance_metrics.items():
            display_val = f"{val:.4f}" if val is not None else "N/A"
            metrics_html += f'<div class="metric-card"><div class="metric-value">{display_val}</div><div class="metric-label">{name.upper()}</div></div>\n'

        features_rows = ""
        for feat in feature_list:
            dtype = str(self.X_train[feat].dtype)
            null_rate = f"{self.X_train[feat].isnull().mean():.2%}"
            features_rows += f"<tr><td>{feat}</td><td>{dtype}</td><td>{null_rate}</td></tr>\n"

        body = f"""
<h1>Model Card: {self.model_name}</h1>
<div class="section">
    <h2>Model Overview</h2>
    <p><strong>Model Type:</strong> {model_type}</p>
    <p><strong>Number of Features:</strong> {n_features}</p>
    <p><strong>Training Samples:</strong> {len(self.X_train)}</p>
    <p><strong>Test Samples:</strong> {len(self.X_test)}</p>
</div>

<h2>Performance Metrics</h2>
<div>{metrics_html}</div>

<h2>Feature Summary</h2>
<table>
    <tr><th>Feature</th><th>Data Type</th><th>Null Rate</th></tr>
    {features_rows}
</table>

<h2>Intended Use</h2>
<div class="section">
    <p>This model is intended for use in production prediction pipelines. It should be monitored for data drift and re-evaluated periodically.</p>
</div>

<h2>Limitations</h2>
<div class="section">
    <ul>
        <li>Performance may degrade on populations not represented in the training data.</li>
        <li>Feature importance does not imply causation.</li>
        <li>Predictions should be reviewed by domain experts in high-stakes decisions.</li>
    </ul>
</div>

<h2>Ethical Considerations</h2>
<div class="section">
    <p>This model should be evaluated for fairness across protected attributes before deployment. Disparate impact and equalized odds should be within acceptable thresholds.</p>
</div>
"""
        html = self.render_html(f"Model Card: {self.model_name}", body)
        if output_path:
            self._save_report(html, output_path)
        return html

    def generate_prediction_report(
        self,
        instance: pd.Series,
        explanation: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a per-prediction explanation report.

        Args:
            instance: Feature values for the prediction.
            explanation: Explanation dict (from SHAPExplainer or LIMEExplainer).
            output_path: Output file path.

        Returns:
            HTML content.
        """
        feature_rows = ""
        for feat, val in instance.items():
            feature_rows += f"<tr><td>{feat}</td><td>{val}</td></tr>\n"

        contribution_rows = ""
        contributions = explanation.get("feature_contributions", [])
        for c in contributions[:15]:
            css_class = "pass" if c["shap_value"] > 0 else "fail"
            contribution_rows += (
                f"<tr><td>{c['feature']}</td><td>{c['value']}</td>"
                f"<td class='{css_class}'>{c['shap_value']:.4f}</td></tr>\n"
            )

        text_exp = explanation.get("text_explanation", "No text explanation available.")

        body = f"""
<h1>Prediction Explanation Report</h1>
<div class="section">
    <p><strong>Prediction:</strong> {explanation.get('prediction', 'N/A')}</p>
    <p><strong>Base Value:</strong> {explanation.get('base_value', 'N/A')}</p>
</div>

<h2>Natural Language Explanation</h2>
<div class="section"><p>{text_exp}</p></div>

<h2>Input Features</h2>
<table>
    <tr><th>Feature</th><th>Value</th></tr>
    {feature_rows}
</table>

<h2>Feature Contributions</h2>
<table>
    <tr><th>Feature</th><th>Value</th><th>SHAP Contribution</th></tr>
    {contribution_rows}
</table>
"""
        html = self.render_html("Prediction Explanation", body)
        if output_path:
            self._save_report(html, output_path)
        return html

    def generate_global_report(
        self,
        importance_df: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a global model behavior report.

        Args:
            importance_df: Feature importance DataFrame from SHAPExplainer.
            output_path: Output file path.

        Returns:
            HTML content.
        """
        metrics_html = ""
        for name, val in self._performance_metrics.items():
            display_val = f"{val:.4f}" if val is not None else "N/A"
            metrics_html += f'<div class="metric-card"><div class="metric-value">{display_val}</div><div class="metric-label">{name.upper()}</div></div>\n'

        importance_rows = ""
        for _, row in importance_df.iterrows():
            importance_rows += (
                f"<tr><td>{row['rank']}</td><td>{row['feature']}</td>"
                f"<td>{row['mean_abs_shap']:.4f}</td>"
                f"<td>{row.get('cumulative_pct', 'N/A')}</td></tr>\n"
            )

        body = f"""
<h1>Global Model Behavior Report</h1>
<h2>Performance Summary</h2>
<div>{metrics_html}</div>

<h2>Global Feature Importance (SHAP)</h2>
<table>
    <tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th><th>Cumulative %</th></tr>
    {importance_rows}
</table>

<h2>Classification Report</h2>
<pre>{classification_report(self.y_test, self.y_pred, zero_division=0)}</pre>
"""
        html = self.render_html("Global Model Report", body)
        if output_path:
            self._save_report(html, output_path)
        return html

    def generate_fairness_report(
        self,
        fairness_metrics: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a fairness analysis report.

        Args:
            fairness_metrics: Output from FairnessAnalyzer.generate_fairness_metrics.
            output_path: Output file path.

        Returns:
            HTML content.
        """
        threshold = self.config.get("fairness", {}).get("threshold", 0.8)

        body = f"""<h1>Fairness Analysis Report</h1>
<div class="section">
    <p><strong>Model:</strong> {self.model_name}</p>
    <p><strong>Fairness Threshold:</strong> {threshold}</p>
</div>
"""
        for attr, attr_metrics in fairness_metrics.items():
            if attr.startswith("_"):
                continue

            body += f"<h2>Protected Attribute: {attr}</h2>\n"

            if "disparate_impact" in attr_metrics:
                di = attr_metrics["disparate_impact"]
                body += "<h3>Disparate Impact</h3>\n<table><tr><th>Group</th><th>Positive Rate</th></tr>\n"
                for group, rate in di.get("group_rates", {}).items():
                    body += f"<tr><td>{group}</td><td>{rate:.4f}</td></tr>\n"
                ratio = di.get("ratio", 0)
                css = "pass" if ratio >= threshold else "fail"
                body += f"</table>\n<p>Disparate Impact Ratio: <span class='{css}'>{ratio:.4f}</span></p>\n"

            if "demographic_parity" in attr_metrics:
                dp = attr_metrics["demographic_parity"]
                body += "<h3>Demographic Parity</h3>\n<table><tr><th>Group</th><th>Positive Prediction Rate</th></tr>\n"
                for group, rate in dp.get("group_rates", {}).items():
                    body += f"<tr><td>{group}</td><td>{rate:.4f}</td></tr>\n"
                body += f"</table>\n<p>Max Difference: {dp.get('max_difference', 0):.4f}</p>\n"

        body += "\n"
        html = self.render_html("Fairness Analysis", body)
        if output_path:
            self._save_report(html, output_path)
        return html

    def generate_compliance_report(
        self,
        regulation: str = "SR 11-7",
        output_path: Optional[str] = None,
        fairness_metrics: Optional[Dict] = None,
    ) -> str:
        """Generate a regulatory compliance report.

        Produces documentation aligned with OCC SR 11-7, CMS, or other
        regulatory frameworks for model risk management.

        Args:
            regulation: Regulatory framework identifier.
            output_path: Output file path.
            fairness_metrics: Optional fairness analysis results.

        Returns:
            HTML content.
        """
        model_type = type(self.model).__name__
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        metrics_rows = ""
        for name, val in self._performance_metrics.items():
            display_val = f"{val:.4f}" if val is not None else "N/A"
            metrics_rows += f"<tr><td>{name}</td><td>{display_val}</td></tr>\n"

        fairness_section = ""
        if fairness_metrics:
            fairness_section = "<h2>5. Fairness and Bias Assessment</h2>\n"
            fairness_section += "<div class='section'><p>Protected attributes were evaluated for disparate impact and demographic parity.</p></div>\n"

        body = f"""
<h1>Model Compliance Report</h1>
<div class="section">
    <p><strong>Regulatory Framework:</strong> {regulation}</p>
    <p><strong>Report Date:</strong> {timestamp}</p>
    <p><strong>Model Name:</strong> {self.model_name}</p>
</div>

<h2>1. Model Description</h2>
<div class="section">
    <p><strong>Algorithm:</strong> {model_type}</p>
    <p><strong>Purpose:</strong> Production prediction model for risk assessment and decision support.</p>
    <p><strong>Training Data Size:</strong> {len(self.X_train)} observations, {len(self.X_train.columns)} features</p>
    <p><strong>Evaluation Data Size:</strong> {len(self.X_test)} observations</p>
</div>

<h2>2. Model Performance</h2>
<table>
    <tr><th>Metric</th><th>Value</th></tr>
    {metrics_rows}
</table>

<h2>3. Model Governance</h2>
<div class="section">
    <ul>
        <li><strong>Model Owner:</strong> [To be assigned]</li>
        <li><strong>Validation Status:</strong> Pending independent validation</li>
        <li><strong>Review Frequency:</strong> Quarterly</li>
        <li><strong>Monitoring:</strong> Monthly performance tracking with automated drift detection</li>
    </ul>
</div>

<h2>4. Risk Assessment</h2>
<div class="section">
    <p><strong>Model Risk Tier:</strong> [To be determined based on materiality assessment]</p>
    <ul>
        <li>Performance degradation monitored via PSI and CSI metrics</li>
        <li>Input data quality validated before each prediction batch</li>
        <li>Fallback procedures documented for model failure scenarios</li>
    </ul>
</div>

{fairness_section}

<h2>{"6" if fairness_metrics else "5"}. Explainability</h2>
<div class="section">
    <p>This model supports per-prediction explanations via SHAP and LIME. Feature attributions are available for every prediction, enabling adverse action notice compliance and audit trail requirements.</p>
</div>

<h2>{"7" if fairness_metrics else "6"}. Limitations and Assumptions</h2>
<div class="section">
    <ul>
        <li>Model assumes feature distributions remain stationary between training and deployment.</li>
        <li>Performance estimates are based on a single holdout test set.</li>
        <li>Explanations are approximate and may not capture all model behavior.</li>
        <li>Model should not be used as the sole basis for consequential decisions without human review.</li>
    </ul>
</div>

<h2>Approval Signatures</h2>
<table>
    <tr><th>Role</th><th>Name</th><th>Signature</th><th>Date</th></tr>
    <tr><td>Model Developer</td><td></td><td></td><td></td></tr>
    <tr><td>Model Validator</td><td></td><td></td><td></td></tr>
    <tr><td>Business Owner</td><td></td><td></td><td></td></tr>
    <tr><td>Risk Officer</td><td></td><td></td><td></td></tr>
</table>
"""
        html = self.render_html(f"Compliance Report -- {regulation}", body)
        if output_path:
            self._save_report(html, output_path)
        return html

    def export_pdf(self, html_content: str, output_path: str) -> str:
        """Export HTML content to PDF.

        Attempts to use weasyprint for conversion. Falls back to saving
        the HTML with a .pdf note if weasyprint is not available.

        Args:
            html_content: Complete HTML document string.
            output_path: PDF output file path.

        Returns:
            The output file path.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            from weasyprint import HTML as WeasyHTML
            WeasyHTML(string=html_content).write_pdf(output_path)
            logger.info("PDF exported to %s", output_path)
        except ImportError:
            html_fallback = output_path.replace(".pdf", ".html")
            with open(html_fallback, "w") as fh:
                fh.write(html_content)
            logger.warning("weasyprint not available; saved as HTML: %s", html_fallback)
            return html_fallback

        return output_path
