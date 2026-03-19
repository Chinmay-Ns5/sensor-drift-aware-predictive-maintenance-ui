import joblib
import pandas as pd
import numpy as np
import warnings

# ------------------------------------------------------------------
# Suppress non-actionable sklearn warnings
# ------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names"
)

# ------------------------------------------------------------------
# Load trained model & feature schema
# ------------------------------------------------------------------
MODEL_PATH = "artifacts/gb_model.joblib"
FEATURES_PATH = "artifacts/feature_cols.joblib"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)

# ------------------------------------------------------------------
# Load test dataset
# ------------------------------------------------------------------
columns = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

df_test = pd.read_csv(
    "CMAPSSData/test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# ------------------------------------------------------------------
# Drop useless sensors (same as training)
# ------------------------------------------------------------------
DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16",
    "sensor_18", "sensor_19"
]

df_test.drop(columns=DROP_SENSORS, inplace=True)

# ------------------------------------------------------------------
# Feature engineering (MUST match training)
# ------------------------------------------------------------------
WINDOW = 20
SENSOR_COLS = [c for c in df_test.columns if c.startswith("sensor_")]

for sensor in SENSOR_COLS:
    df_test[f"{sensor}_roll_mean"] = (
        df_test.groupby("engine_id")[sensor]
        .transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
    )

    df_test[f"{sensor}_roll_std"] = (
        df_test.groupby("engine_id")[sensor]
        .transform(lambda x: x.rolling(WINDOW, min_periods=1).std())
    )

    df_test[f"{sensor}_trend"] = (
        df_test.groupby("engine_id")[sensor]
        .transform(lambda x: x.diff(WINDOW))
    )

df_test.fillna(0, inplace=True)

# ------------------------------------------------------------------
# Risk categorization
# ------------------------------------------------------------------
def risk_category(prob):
    if prob < 0.2:
        return "LOW RISK"
    elif prob < 0.5:
        return "MEDIUM RISK"
    elif prob < 0.75:
        return "HIGH RISK"
    else:
        return "CRITICAL RISK"

# ------------------------------------------------------------------
# Uncertainty estimation (bootstrap-style)
# ------------------------------------------------------------------
def predict_with_uncertainty(X, model, n_samples=50):
    probs = []
    for _ in range(n_samples):
        probs.append(model.predict_proba(X)[0, 1])
    return np.mean(probs), np.std(probs)

# ------------------------------------------------------------------
# Predict failure risk for ONE engine
# ------------------------------------------------------------------
def predict_engine(engine_id):
    engine_df = df_test[df_test["engine_id"] == engine_id]

    if engine_df.empty:
        return None, None

    latest_row = engine_df.sort_values("cycle").iloc[-1]

    X = pd.DataFrame(
        [latest_row[feature_cols].values],
        columns=feature_cols
    )

    mean_risk, std_risk = predict_with_uncertainty(X, model)
    return mean_risk, std_risk

# ------------------------------------------------------------------
# Predict failure risk for RANGE of engines (fleet ranking)
# ------------------------------------------------------------------
def predict_engine_range(start_id, end_id, threshold=0.5):
    results = []

    for eid in range(start_id, end_id + 1):
        mean_risk, std_risk = predict_engine(eid)
        if mean_risk is not None:
            results.append({
                "Engine ID": eid,
                "Failure Risk": mean_risk,
                "Uncertainty": std_risk
            })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    # Fleet-relative ranking
    df["Risk Rank"] = df["Failure Risk"].rank(
        ascending=False, method="dense"
    ).astype(int)

    df["Risk Category"] = df["Failure Risk"].apply(risk_category)
    df["Status"] = df["Failure Risk"].apply(
        lambda x: "ALERT" if x >= threshold else "OK"
    )

    return df.sort_values("Risk Rank")

# ------------------------------------------------------------------
# Command Line Interface
# ------------------------------------------------------------------
if __name__ == "__main__":

    print("\n=== Predictive Maintenance System ===")
    print("1. Predict single engine")
    print("2. Predict range of engines")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        eid = int(input("Enter Engine ID: "))
        mean_risk, std_risk = predict_engine(eid)

        if mean_risk is None:
            print("\nEngine ID not found.")
        else:
            print("\n--- Prediction Result ---")
            print(f"Engine ID      : {eid}")
            print(f"Failure Risk   : {mean_risk:.3f} ± {std_risk:.3f}")
            print(f"Risk Category  : {risk_category(mean_risk)}")

    elif choice == "2":
        start_id = int(input("Enter start Engine ID: "))
        end_id = int(input("Enter end Engine ID: "))
        threshold_input = input("Enter alert threshold (default 0.5): ").strip()
        threshold = float(threshold_input) if threshold_input else 0.5

        results_df = predict_engine_range(start_id, end_id, threshold)

        print("\n--- Engine Risk Summary ---")
        if results_df.empty:
            print("No valid engine IDs found.")
        else:
            print(
                results_df.assign(
                    **{
                        "Failure Risk": results_df["Failure Risk"].round(4),
                        "Uncertainty": results_df["Uncertainty"].round(4)
                    }
                ).to_string(index=False)
            )

    else:
        print("\nInvalid choice. Please run again.")
