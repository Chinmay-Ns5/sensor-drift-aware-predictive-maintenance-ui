# Sensor Drift-Aware Predictive Maintenance — Interactive Dashboard

## Overview

This repository extends the
[sensor-drift-aware-predictive-maintenance-ml](https://github.com/Chinmay-Ns5/sensor-drift-aware-predictive-maintenance-ml)
project with a **full Streamlit dashboard**, providing an interactive
interface for fleet monitoring, engine lifecycle analysis, sensor drift
detection, and live inference from manual sensor input.

The underlying model, feature engineering pipeline, and inference logic
remain identical to the base project. This repository adds a
deployment-ready UI layer on top.

------------------------------------------------------------------------

## Problem Statement

Given sensor readings collected over time from multiple engines, the
objective is to:

-   Predict whether an engine is likely to fail within a predefined
    horizon
-   Prioritize engines based on **relative risk**
-   Ensure predictions remain reliable under **sensor drift**
-   Provide **interpretable and actionable outputs**
-   Visualise all of the above in a real-time monitoring dashboard

------------------------------------------------------------------------

## Dashboard Pages

### 📡 Fleet Dashboard

Two modes for fleet-level analysis:

-   **Test Mode (Deployment)** — Snapshot of current risk across all
    engines based on latest sensor readings. Colour-coded bar chart
    with adjustable alert threshold. Summary cards showing total,
    critical, on-alert, and healthy engine counts.

-   **Train Mode (Lifecycle)** — Select one or more engines and plot
    their full risk progression from cycle 1 to failure. Visualises
    how predicted risk evolves as engines degrade. Per-engine summary
    cards with automated fleet insights — longest life, shortest life,
    most gradual degradation, and cycle-level risk escalation point.

### 🔍 Engine Deep Dive

-   **Dataset Lookup** — Select any engine ID. Displays risk score,
    uncertainty range, risk gauge, sensor trend charts (raw + rolling
    mean), and top 15 predictive features by model importance.

-   **Manual Sensor Input** — Enter raw sensor readings directly into
    input fields. The system applies the full feature engineering
    pipeline (rolling mean, std, trend over window = 20 cycles) and
    runs live inference — simulating a real-time IoT sensor feed.
    Includes an expandable panel showing every computed feature value
    for full transparency.

### 📡 Sensor Drift Monitor

-   PSI (Population Stability Index) bar chart per sensor
-   KS test statistic and p-value for each sensor
-   Interactive distribution overlay — early healthy cycles vs
    near-failure cycles
-   Automated interpretation: all sensors show HIGH DRIFT because the
    comparison is intentionally between opposite engine states

### 📖 How It Works

Full walkthrough of the ML pipeline, feature engineering with a worked
example, risk category definitions, and a page-by-page usage guide.

------------------------------------------------------------------------

## End-to-End Workflow

    Raw Sensor Data (C-MAPSS)
            │
            ▼
    Preprocessing
    - Remove constant / near-constant sensors
    - Preserve engine-wise temporal order
            │
            ▼
    Feature Engineering
    - Rolling mean (local behavior)
    - Rolling standard deviation (instability)
    - Trend (rate of degradation)
            │
            ▼
    Label Construction
    - Compute Remaining Useful Life (RUL)
    - Convert RUL to failure labels
            │
            ▼
    Validation Strategy
    - Engine-aware temporal split
    - Prevents future data leakage
            │
            ▼
    Model Training
    - Logistic Regression (baseline)
    - Random Forest
    - Gradient Boosting (selected model)
            │
            ▼
    Cost-Sensitive Decision Logic
    - Asymmetric FN / FP costs
    - Threshold tuning
            │
            ▼
    Interpretability & Monitoring
    - SHAP explanations
    - Sensor drift detection (PSI, KS test)
            │
            ▼
    Deployment
    - Serialized model artifacts
    - Standalone CLI inference tool
    - Fleet-level risk ranking
    - Interactive Streamlit dashboard

------------------------------------------------------------------------

## Model Results

| Model | ROC-AUC | Recall (Fail) | Precision (Fail) | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.9925 | 0.9597 | 0.7850 | 0.8636 |
| Random Forest | 0.9891 | 0.8548 | 0.9060 | 0.8797 |
| **Gradient Boosting** | **0.9899** | **0.8855** | **0.8927** | **0.8891** |

Gradient Boosting selected as final model. Recall is prioritised over
accuracy — in predictive maintenance, missing a real failure (false
negative) is far more costly than a false alarm (false positive).

------------------------------------------------------------------------

## Dataset Description

**NASA C-MAPSS FD001**

-   `train_FD001.txt`\
    Full run-to-failure sensor data.\
    Used for feature engineering, training, validation,
    interpretability, and drift analysis.

-   `test_FD001.txt`\
    Truncated sensor streams (engines not yet failed).\
    Used **only for deployment-style inference**.

> Test data is never used during training, ensuring zero data leakage.

------------------------------------------------------------------------

## Feature Engineering

Raw sensor values are transformed into degradation-aware features:

-   **Rolling Mean** — smooth operational behavior
-   **Rolling Standard Deviation** — instability growth
-   **Trend (difference over window)** — degradation rate

All features are computed **per engine**, preserving temporal integrity.\
14 active sensors × 3 features = **42 model inputs**.

------------------------------------------------------------------------

## Sensor Drift Detection

Sensor drift is monitored using:

-   **Population Stability Index (PSI)** — magnitude of distribution
    shift
-   **Kolmogorov–Smirnov (KS) test** — statistical significance

Reference = early cycles (cycle ≤ 50)\
Current = near-failure cycles (last 50 cycles of each engine's life)

All sensors show HIGH DRIFT and KS p-value = 0.0. This is expected and
correct — engines are compared at physically opposite states (healthy
vs about to fail). High PSI = high degradation signal = high predictive
power.

------------------------------------------------------------------------

## Project Structure

    artifacts/            # Saved model and feature schema
    CMAPSSData/           # NASA C-MAPSS dataset
    app.py                # Streamlit dashboard (4 pages)
    main.py               # Deployment-ready CLI inference script
    requirements.txt      # Python dependencies
    README.md             # Project documentation

------------------------------------------------------------------------

## Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
joblib>=1.3.0
scipy>=1.11.0
```

------------------------------------------------------------------------

## How to Run

**Install dependencies**

``` bash
pip install -r requirements.txt
```

**Launch the dashboard**

``` bash
streamlit run app.py
```

**Or use the CLI tool**

``` bash
python main.py
```

Follow the CLI prompts to predict risk for a single engine or rank risk
across a fleet.

------------------------------------------------------------------------

## Example Output (CLI)

    Engine ID | Failure Risk | Risk Rank | Risk Category | Status
    -------------------------------------------------------------
    3         | 0.0533       | 1         | LOW RISK      | OK
    5         | 0.0062       | 2         | LOW RISK      | OK
    8         | 0.0061       | 3         | LOW RISK      | OK

This reflects realistic deployment behavior where most engines are
healthy, but relative risk is still prioritized.

------------------------------------------------------------------------

## Key Takeaways

-   Emphasis on **ML correctness and system design**
-   Temporal validation to prevent data leakage
-   Classical ML chosen for robustness and interpretability
-   Deployment-ready pipeline with both CLI and interactive UI
-   Sensor drift monitoring with statistical rigor (PSI + KS test)
-   Manual sensor input simulates real-time IoT inference

------------------------------------------------------------------------

## Related Repository

Full analysis notebook (training, SHAP interpretability, drift
analysis):
[sensor-drift-aware-predictive-maintenance-ml](https://github.com/Chinmay-Ns5/sensor-drift-aware-predictive-maintenance-ml)
