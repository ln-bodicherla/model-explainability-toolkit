# Model Explainability Toolkit

Comprehensive model interpretability using SHAP and LIME. Generates per-prediction explanations, global feature importance rankings, and compliance-ready reports. Built for regulated industries including finance, healthcare, and insurance.

## Features

- **SHAP Explanations** -- TreeExplainer, KernelExplainer, and DeepExplainer with automatic model type detection
- **LIME Explanations** -- Local interpretable model-agnostic explanations for tabular and text data
- **Global Feature Importance** -- Mean absolute SHAP values, feature interaction analysis, beeswarm plots
- **Per-Prediction Reports** -- Waterfall plots, force plots, and natural language explanations for individual predictions
- **Fairness Analysis** -- Disparate impact, equalized odds, demographic parity, and calibration metrics across protected attributes
- **Compliance Reports** -- Regulatory-ready documentation for CMS, SR 11-7, and internal audit requirements
- **Model Cards** -- Standardized model documentation with performance, fairness, and limitation disclosures
- **REST API** -- FastAPI service for on-demand explanations

## Architecture

```
model-explainability-toolkit/
├── src/
│   ├── shap_explainer.py       # SHAP-based explanations
│   ├── lime_explainer.py       # LIME-based explanations
│   ├── report_generator.py     # HTML/PDF report generation
│   ├── fairness_analyzer.py    # Bias and fairness metrics
│   └── api.py                  # FastAPI REST endpoints
├── configs/
│   └── config.yaml             # Explainability configuration
├── templates/                  # Jinja2 report templates
├── reports/                    # Generated reports (gitignored)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/ln-bodicherla/model-explainability-toolkit.git
cd model-explainability-toolkit
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### SHAP Explanations

```python
from src.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model=trained_model, X_train=X_train)
explainer.setup_explainer()

# Single prediction explanation
explanation = explainer.explain_prediction(X_test.iloc[0])
print(explanation["text_explanation"])

# Global feature importance
importance = explainer.global_feature_importance(X_test)

# Visualizations
explainer.plot_waterfall(X_test.iloc[0], save_path="waterfall.png")
explainer.plot_beeswarm(X_test, save_path="beeswarm.png")
```

### LIME Explanations

```python
from src.lime_explainer import LIMEExplainer

explainer = LIMEExplainer(model=trained_model, X_train=X_train)
explainer.setup_explainer()

explanation = explainer.explain_prediction(X_test.iloc[0])
top_features = explainer.get_top_features(X_test.iloc[0], n=5)
```

### Fairness Analysis

```python
from src.fairness_analyzer import FairnessAnalyzer

analyzer = FairnessAnalyzer(
    y_true=y_test,
    y_pred=model.predict(X_test),
    y_proba=model.predict_proba(X_test)[:, 1],
    protected_attributes=X_test[["gender", "age_group"]],
)

metrics = analyzer.generate_fairness_metrics()
analyzer.plot_fairness_dashboard(save_path="fairness_dashboard.png")
```

### Compliance Reports

```python
from src.report_generator import ExplainabilityReportGenerator

generator = ExplainabilityReportGenerator(
    model=trained_model,
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
)

generator.generate_compliance_report(
    regulation="SR 11-7",
    output_path="reports/compliance_report.html",
)
```

### REST API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000

curl -X POST "http://localhost:8000/explain/shap" \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 45, "income": 75000, "credit_score": 720}}'
```

## Configuration

All parameters are centralized in `configs/config.yaml`:

| Section | Parameter | Description | Default |
|---|---|---|---|
| `shap.max_samples` | Background dataset size | `1000` |
| `lime.num_features` | Features in LIME explanation | `10` |
| `lime.num_samples` | LIME perturbation samples | `5000` |
| `fairness.threshold` | Minimum acceptable ratio | `0.8` |
| `reporting.format` | Output report format | `html` |

## Regulatory Compliance

This toolkit supports documentation requirements for:

- **OCC SR 11-7** -- Model risk management guidance for banking organizations
- **CMS** -- Centers for Medicare & Medicaid Services model transparency requirements
- **GDPR Article 22** -- Right to explanation for automated decision-making
- **ECOA / Reg B** -- Adverse action notice requirements with specific reason codes

## Requirements

- Python 3.9+
- scikit-learn compatible model (or any model with `predict` / `predict_proba`)

## License

MIT License
